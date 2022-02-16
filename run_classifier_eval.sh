CURRENT_DIR=`pwd`
export BERT_BASE_DIR=$CURRENT_DIR/bert_base_uncased
export GLUE_DIR=$CURRENT_DIR/data
export OUTPUR_DIR=$CURRENT_DIR/outputs
TASK_NAME="SST"

python run_classifier.py \
  --model_type=bert \
  --model_name_or_path=$BERT_BASE_DIR \
  --task_name=$TASK_NAME \
  --do_eval \
  --do_kd \
  --do_lower_case \
  --data_dir=$GLUE_DIR \
  --max_seq_length=128 \
  --per_gpu_train_batch_size=16 \
  --per_gpu_eval_batch_size=16 \
  --learning_rate=2e-5 \
  --num_train_epochs=6.0 \
  --logging_steps=500 \
  --save_steps=500 \
  --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
  --overwrite_output_dir \
  --seed=42 \
  --kd_coeff=0.45 \
  --eval_all_checkpoints
