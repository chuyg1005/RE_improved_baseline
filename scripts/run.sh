export CUDA_VISIBLE_DEVICES=$1;
DATASET=$2;
SPLIT=origin;
MODE=MixDebias;
TRAIN_NAME=train4debias2;

SEED=42;
python train_tacred.py --model_name bert-base-cased \
  --split $SPLIT \
  --seed $SEED --dataset $DATASET \
  --mode $MODE --train_name $TRAIN_NAME \
  --gradient_accumulation_steps 2 --train_batch_size 32;

#python evaluation.py --model_name roberta-base --input_format typed_entity_marker_punct --seed $SEED --dataset tacmock --eval_name test;
#python gen_challege_dataset.py --model_name bert-base-cased --dataset tacred --eval_name test --seed 0;
