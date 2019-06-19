nvidia-smi
MODEL=search_cifar10
OUTPUT_DIR=exp/$MODEL
DATA_DIR=data

mkdir -p $OUTPUT_DIR

python train_search.py \
  --data=$DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --child_layers=3 \
  --child_eval_epochs=50 \
  --child_epochs=200 \
  --child_batch_size=64 \
  --child_keep_prob=1.0 \
  --child_drop_path_keep_prob=0.9 \
  --child_sample_policy='params' \
  --controller_expand=8 \
  --controller_seed_arch=600 | tee -a $OUTPUT_DIR/train.log