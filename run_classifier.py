from __future__ import absolute_import, division, print_function
import time
import argparse
import glob
import logging
import os
import json
import copy
import numpy as np
import torch
from tqdm import tqdm
from torch.nn import MSELoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from models.transformers import AdamW, WarmupLinearSchedule
from models.transformers import (WEIGHTS_NAME, BertConfig,BertForSequenceClassification, BertTokenizer)

from processors.tasks_compute_metrics import compute_metrics
from processors.load_and_cache import load_and_cache_data
from processors.common import seed_everything, save_numpy
from processors.common import init_logger, logger
from processors import task_output_modes as output_modes 
from processors import task_processors as processors
from processors import task_convert_examples_to_features as convert_examples_to_features
from processors import collate_fn, xlnet_collate_fn
import warnings
warnings.filterwarnings("ignore")

# load BERT configuration file and tokenizer
pretrained_model = BertConfig.pretrained_config_archive_map.keys()
models = {'bert': (BertConfig, BertForSequenceClassification, BertTokenizer)}

def train(args, train_dataset, model, tokenizer):
    '''
    model training
    :param model: pretrained model
    :param train_dataset:
    :param tokenizer: 
    '''
    # determine the training batch size
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)  
    else:
        DistributedSampler(train_dataset)

    # create data loader
    train_dataloader = DataLoader(train_dataset, 
                                  sampler=train_sampler, 
                                  batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)

    # determine total optimization and warm up steps 
    ga_steps = args.gradient_accumulation_steps
    num_train_data = len(train_dataloader)
    if args.max_steps > 0:
        total_steps = args.max_steps
        args.num_train_epochs = args.max_steps // num_train_data// ga_steps) + 1
    else:
        total_steps = num_train_data // ga_steps * args.num_train_epochs
    args.warmup_steps = int(total_steps * args.warmup_proportion)
    print("===Warm up steps===: ",args.warmup_steps)

    # initialize optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    decay_params = [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]
    nodecay_params = [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)]

    optimizer_grouped_parameters = [
        {'params': decay_params,'weight_decay': args.weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=total_steps)
    
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi gpu
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # distributed training
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, 
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        find_unused_parameters=True)
    
    # calculate total train batch size
    # with parallel, distributed and accumulation
    total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * (
                torch.distributed.get_world_size() if args.local_rank != -1 else 1)
    
    # Begin Training
    logger.info("======== Running training ========")
    logger.info("  Number of epochs = %d", args.num_train_epochs)
    logger.info("  Number of records = %d", len(train_dataset))
    logger.info("  Instantaneous batch size per {} = {}".format(
                    args.device,args.per_gpu_train_batch_size))
    logger.info("  Total training batch size = %d",total_train_batch_size)
    logger.info("  Gradient accumulation steps = %d", ga_steps)
    logger.info("  Total optimization steps = %d", total_steps)

    # initialize var and seed everything
    seed_everything(args.seed) 
    global_step = 0
    loss_training = 0.0
    loss_logging = 0.0
    model.zero_grad()
    loss_curve = []
    best_acc = -100

    ############### Knowledge Distillation ###########
    # if running kd then create a student (kd_model)
    # note there is on backward for kd model 
    # only mannully updates the weights

    if args.do_kd:
        kd_loss_func = MSELoss()
        kd_model = copy.deepcopy(model)
        kd_model.eval()

    for current_epoch in range(int(args.num_train_epochs)):
        for step, batch in enumerate(tqdm(train_dataloader,
                                total=num_train_data, 
                                desc='Training',ncols=120,
                                position=0, leave=True)):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            # define model inputs, no segment_ids for DistilBERT 
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[3]}
            if args.model_type in ['bert']:
                token_type = batch[2] 
            else:
                None 
            if args.model_type != 'distilbert':
                inputs['token_type_ids'] = token_type
            
            # get loss of teacher model
            outputs = model(**inputs)
            loss = outputs[0] 

            if args.do_kd:
                # set no label for kd model(student)
                inputs['labels'] = None
                with torch.no_grad():
                    outputs_kd = kd_model(**inputs)
                kd_loss = kd_loss_func(outputs[1], outputs_kd[0])
                loss_curve.append([loss.mean().item(),kd_loss.item()])
                loss += args.kd_coeff * kd_loss
            
            # average on multi-gpu parallel training
            if args.n_gpu > 1:
                loss = loss.mean()  
            if ga_steps > 1:
                loss = loss / ga_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
           
            loss_training += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                if args.do_kd:
                    decay = min(args.decay, (1 + global_step) / (10 + global_step))
                    with torch.no_grad():
                        parameters = [p for p in model.parameters() if p.requires_grad]
                        for kd_param, param in zip(kd_model.parameters(), parameters):
                            # mannully update kd model's parameters 
                            kd_param.sub_((1.0 - decay) * (kd_param - param))
                if args.local_rank in [-1, 0] and args.logging_steps > 0 
                               and global_step % args.logging_steps == 0:
                    print("\n")
                    # Log metrics when sigle GPU
                    if args.local_rank == -1:  
                        eval_res = evaluate(args, model, tokenizer)
                        # current epoch accuracy
                        cur_acc = eval_res['acc'] 
                        if cur_acc > best_acc:
                            best_acc = cur_acc
                            # save model checkpoint
                            output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            if hasattr(model,'module'):
                                model_to_save = model.module 
                            else: 
                                model_to_save = model  
                            model_to_save.save_pretrained(output_dir)
                            torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                            logger.info("Saving model checkpoint to %s", output_dir)
                            tokenizer.save_vocabulary(vocab_path=output_dir)
        print(" ")
        with open('loss_curve.txt', 'w') as f:
            for item in loss_curve:
                f.write("%s\n" % item)
        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()
    return global_step, loss_training / global_step

def evaluate(args, model, tokenizer, prefix=""):
    '''
    evaluate model
    :param model: model to evaluate
    :param tokenizer: model tokenizer
    :param prefix: checkpoint.split('/')[-1] 
    '''
    eval_tasks = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)
    results = {}
    for task, output_dir in zip(eval_tasks, eval_outputs_dirs):
        eval_dataset = load_and_cache_data(args, task, tokenizer, data_type='dev')
        if not os.path.exists(output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(output_dir)
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        if args.local_rank == -1:
            sampler_to_use = SequentialSampler(eval_dataset)  
        else:
            sampler_to_use = DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=sampler_to_use, 
                                     batch_size=args.eval_batch_size,
                                     collate_fn=collate_fn)
        # Begin Evaluation
        logger.info("======== Running evaluation {} ========".format(prefix))
        eval_loss = 0.0
        eval_steps = 0
        preds = None
        out_label_ids = None
        for step, batch in enumerate(tqdm(eval_dataloader,
                                    total=len(eval_dataloader), 
                                    desc="Evaluating",ncols=120,
                                    position=0, leave=True)):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'labels': batch[3]}

                if args.model_type in ['bert']:
                    token_type = batch[2] 
                else:
                    None 
                if args.model_type != 'distilbert':
                    inputs['token_type_ids'] = token_type
                
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()
            eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        print(' ')
        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()
        eval_loss = eval_loss / eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(task, preds, out_label_ids)
        results.update(result)
        logger.info("  Number of examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        logger.info("======== Evaluation results {} ========".format(prefix))
        for key in sorted(result.keys()):
            logger.info(" dev: %s = %s", key, str(result[key]))
    return results



def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--data_dir", default='D:/my/SST-2', type=str, required=False,
                        help="Input data directory that contain the .tsv files.")
    parser.add_argument("--model_type", default='bert', type=str, required=False,
                        help="Model type selected in the list: " + ", ".join(models.keys()))
    parser.add_argument("--model_name_or_path", default='D:/pretrain_model/bert_base_uncased', type=str, required=False,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                            pretrained_model))
    parser.add_argument("--task_name", default="SST", type=str, required=False,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--output_dir", default='./outputs', type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be saved.")

    # Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action='store_true',
                        help="Whether to run the model in inference mode on the test set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--do_kd", action='store_true',
                        help="Whether to do knowledge distillation (KD).")
    parser.add_argument("--kd_coeff", type=float,default=0.9,
                        help="KD loss coefficient.")
    parser.add_argument("--decay",default=0.995,type=float,help='The exponential decay')

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training.")

    parser.add_argument('--logging_steps', type=int, default=3,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=1000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--predict_checkpoints", type=int, default=0,
                        help="predict checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    args = parser.parse_args()
    
    # check if GPU is available
    if torch.cuda.is_available():
        cur_device = 'cuda'
    else:
        cur_device = 'cpu'
    args.device = torch.device(cur_device)

    # check if output directory exist
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    dir_temp = list(filter(None, args.model_name_or_path.split('/'))).pop()
    args.output_dir = args.output_dir + '{}'.format(dir_temp)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    
    init_logger(log_file=args.output_dir + '/my.log')
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome."
                        .format(args.output_dir))

    # distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - 
        # https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # set up cuda
    if torch.cuda.is_available() and not args.no_cuda:
        cur_device = "cuda"
    else:
        cur_device = "cpu"
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(cur_device)
        args.n_gpu = torch.cuda.device_count()
    else:  # initialize the distributed backend 
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # setup logging
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)
    # seed
    seed_everything(args.seed)
    # prepare task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # load pretrained model and tokenizer
    ## only the first process in distributed training will download model & vocab
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = models[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels, finetuning_task=args.task_name)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config)
    if args.local_rank == 0:
        torch.distributed.barrier()  
    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)
    
    # start training
    if args.do_train:
        train_dataset = load_and_cache_data(args, args.task_name, tokenizer, data_type='train')
        global_step, loss_training = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, loss_training)

    # saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in 
                               sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, 
                               recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            if checkpoint.find('checkpoint') != -1:
                prefix = checkpoint.split('/')[-1]  
            else:
                prefix = ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)
        output_eval_file = os.path.join(args.output_dir, "checkpoint_eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("%s = %s\n" % (key, str(results[key])))

if __name__ == "__main__":
    main()
