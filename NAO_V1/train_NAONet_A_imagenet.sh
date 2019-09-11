# Train the architecture discovered by NAO, noted as NAONet-A, on ImageNet, with single card
nvidia-smi
MODEL=NAONet_A_imagenet
OUTPUT_DIR=exp/$MODEL
DATA_DIR=imagenet/raw-data

mkdir -p $OUTPUT_DIR

fixed_arc="0 7 1 15 2 8 1 7 0 6 1 7 0 8 4 7 1 5 0 7 0 7 1 13 0 6 0 14 0 9 1 10 0 14 2 6 1 11 0 7"

python train_imagenet.py \
  --data=$DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --batch_size=128 \
  --arch="$fixed_arc" \
  --channels=42 \
  --use_aux_head \
  --lr=0.1 | tee -a $OUTPUT_DIR/train.log
