# Train the architecture discovered by NAO-V2, noted as NAONet-V2, on ImageNet, with single card
nvidia-smi
MODEL=NAONet_V2_imagenet
OUTPUT_DIR=exp/$MODEL
DATA_DIR=imagenet/raw-data

mkdir -p $OUTPUT_DIR

fixed_arc="0 0 0 3 0 1 2 1 0 2 0 0 0 3 0 0 0 0 0 0 0 0 0 0 1 4 0 1 2 1 0 2 0 0 0 3 0 0 5 0"

python train_imagenet.py \
  --data=$DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --batch_size=128 \
  --arch="$fixed_arc" \
  --use_aux_head \
  --channels=52 \
  --lr=0.1 | tee -a $OUTPUT_DIR/train.log
