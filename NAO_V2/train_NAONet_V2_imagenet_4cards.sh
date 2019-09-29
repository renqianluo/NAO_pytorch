# Train the architecture discovered by NAO-V2, noted as NAONet-V2, on ImageNet, with 4 cards
nvidia-smi
MODEL=NAONet_V2_imagenet
OUTPUT_DIR=exp/$MODEL
DATA_DIR=imagenet/raw-data

mkdir -p $OUTPUT_DIR

fixed_arc="1 1 0 1 2 4 0 0 0 0 2 1 0 0 0 2 0 3 0 0 0 2 0 2 2 2 1 0 0 0 0 1 0 0 0 3 0 0 0 1"

python train_imagenet.py \
  --data=$DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --batch_size=512 \
  --arch="$fixed_arc" \
  --use_aux_head \
  --channels=52 \
  --lr=0.4 | tee -a $OUTPUT_DIR/train.log
