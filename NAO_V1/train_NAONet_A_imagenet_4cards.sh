# Train the architecture discovered by NAO, noted as NAONet-A, on ImageNet, with 4 cards
nvidia-smi
MODEL=NAONet_A_imagenet
OUTPUT_DIR=exp/$MODEL
DATA_DIR=imagenet/raw-data

mkdir -p $OUTPUT_DIR

fixed_arc="0 2 1 10 2 3 1 2 0 1 1 2 0 3 4 2 1 0 0 2 0 2 1 8 0 1 0 9 0 4 1 5 0 9 2 1 1 6 0 2"

python train_imagenet.py \
  --data=$DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --search_space=large \
  --batch_size=512 \
  --arch="$fixed_arc" \
  --use_aux_head \
  --channels=42 \
  --lr=0.4 | tee -a $OUTPUT_DIR/train.log
