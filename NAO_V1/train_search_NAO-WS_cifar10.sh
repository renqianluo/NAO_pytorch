MODEL=search_cifar10
OUTPUT_DIR=exp/$MODEL
DATA_DIR=data

mkdir -p $OUTPUT_DIR

python train_search.py --data=$DATA_DIR --output_dir=$OUTPUT_DIR | tee -a $OUTPUT_DIR/train.log
