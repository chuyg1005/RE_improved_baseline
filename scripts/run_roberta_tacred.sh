#for SEED in 78 23 61;
#do python train_tacred.py --model_name_or_path roberta-large --input_format typed_entity_marker_punct --seed $SEED --run_name roberta;
#done;

export CUDA_VISIBLE_DEVICE=$1;
SEED=0;
DATASET=$2;
INPUT_FORMAT=$3;
python train_tacred.py --model_name bert-base-cased --input_format $INPUT_FORMAT --seed $SEED --dataset $DATASET;
#python evaluation.py --model_name roberta-base --input_format typed_entity_marker_punct --seed $SEED --dataset tacmock --eval_name test;