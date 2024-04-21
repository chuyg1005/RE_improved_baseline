DEVICE=$1;
DATASET=$2;
export CUDA_VISIBLE_DEVICES=$DEVICE;
python gen_challege_dataset.py --dataset $DATASET --version v0 --mode default;
python gen_challege_dataset.py --dataset $DATASET --version v1;
