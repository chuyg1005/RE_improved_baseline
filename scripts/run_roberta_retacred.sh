#for SEED in 78 23 61;
#do python train_retacred.py --model_name_or_path roberta-large --input_format typed_entity_marker_punct --seed $SEED --run_name roberta;
#done;

SEED=0
python train_retacred.py --model_name roberta-large --input_format typed_entity_marker_punct --seed $SEED --run_name roberta --num_class 40;
