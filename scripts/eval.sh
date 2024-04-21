export CUDA_VISIBLE_DEVICES=$1;
DATASET=$2;
TRAIN_MODE=$3;
EVAL_NAME=$4;
SEED=42;
python evaluation.py --dataset $DATASET --eval_name $EVAL_NAME \
    --mode $TRAIN_MODE --save_predict --seed $SEED;