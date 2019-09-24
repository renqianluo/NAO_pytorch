# Train the architecture discorved by NAO, with channel size of 128, noted as NAONet-A-128
nvidia-smi
MODEL=NAONet_A_128_cifar10
OUTPUT_DIR=exp/$MODEL
DATA_DIR=data

mkdir -p $OUTPUT_DIR

fixed_arc="0 2 1 10 2 3 1 2 0 1 1 2 0 3 4 2 1 0 0 2 0 2 1 8 0 1 0 9 0 4 1 5 0 9 2 1 1 6 0 2"

python train_cifar.py \
  --data=$DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --search_space=large \
  --arch="$fixed_arc" \
  --use_aux_head \
  --channels=128 \
  --cutout_size=16 \
  --l2_reg=5e-4 \
  --keep_prob=0.6 \
  --drop_path_keep_prob=0.7 | tee -a $OUTPUT_DIR/train.log
