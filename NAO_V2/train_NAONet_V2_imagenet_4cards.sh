# Train the architecture discovered by NAO-V2, noted as NAONet-V2, on ImageNet, with 4 cards
nvidia-smi
MODEL=NAONet_V2_imagenet
OUTPUT_DIR=exp/$MODEL
DATA_DIR=imagenet/raw-data

mkdir -p $OUTPUT_DIR

fixed_arc="0 1 0 4 0 1 1 0 2 0 2 4 2 4 2 1 2 0 3 2 1 3 0 4 0 1 1 0 0 1 0 2 2 0 0 1 1 0 1 4"

python train_imagenet.py \
  --data=$DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --batch_size=512 \
  --arch="$fixed_arc" \
  --layers=6 \
  --channels=44 \
  --use_aux_head \
  --lr=0.4 | tee -a $OUTPUT_DIR/train.log
