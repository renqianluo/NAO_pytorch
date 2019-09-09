# Trainthe architecture discovered by NAO-WS, with channel size of 36, noted as NAONet-B-36
nvidia-smi
MODEL=NAONet_B_36_cifar10
OUTPUT_DIR=exp/$MODEL
DATA_DIR=data

mkdir -p $OUTPUT_DIR

fixed_arc="0 1 0 2 0 1 2 2 1 3 0 2 4 0 1 3 2 1 0 3 1 1 1 0 2 1 2 4 1 0 2 3 3 1 2 2 0 2 0 2"

python train_cifar.py \
  --data=$DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --batch_size=128 \
  --arch="$fixed_arc" \
  --channels=36 \
  --use_aux_head \
  --cutout_size=16 | tee -a $OUTPUT_DIR/train.log
