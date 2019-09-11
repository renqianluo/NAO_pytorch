# Train the architecture discorved by NAO, with channel size of 128, noted as NAONet-A-128
nvidia-smi
MODEL=NAONet_A_128_cifar10
OUTPUT_DIR=exp/$MODEL
DATA_DIR=data

mkdir -p $OUTPUT_DIR

fixed_arc="0 7 1 15 2 8 1 7 0 6 1 7 0 8 4 7 1 5 0 7 0 7 1 13 0 6 0 14 0 9 1 10 0 14 2 6 1 11 0 7"

python train_cifar.py \
  --data=$DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --arch="$fixed_arc" \
  --channels=128 \
  --cutout_size=16 \
  --keep_prob=0.6 \
  --drop_path_keep_prob=0.7 | tee -a $OUTPUT_DIR/train.log
