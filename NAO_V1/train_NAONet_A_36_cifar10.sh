# Train the architecture discovered by NAO, with channel size of 36, noted as NAONet-A-36
nvidia-smi
MODEL=NAONet_A_36_cifar10
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
  --cutout_size=16 | tee -a $OUTPUT_DIR/train.log
