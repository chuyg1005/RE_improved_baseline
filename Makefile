DEVICE=1
MODE=default
TRAIN_FILE=train

prepare:
	bash scripts/run_roberta_tacred.sh $DEVICE entity_mask default train
	bash scripts/run_roberta_tacred.sh $DEVICE typed_entity_name_punct default train
	python gen_challenge_dataset.py --dataset tacred --eval_name test

train:
	bash scripts/run_roberta_tacred.sh $DEVICE typed_entity_marker_punct $MODE $TRAIN_FILE

eval:
	python evaluation.py --dataset tacred --eval_name test --mode MODE
	python evaluation.py --dataset tacred --eval_name test_challenge --mode MODE
