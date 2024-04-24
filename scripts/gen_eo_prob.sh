device=$1
dataset=$2
mode=$3

bash scripts/eval.sh $device $dataset $mode dev-eo
bash scripts/eval.sh $device $dataset $mode test-eo

if [ $dataset == "tacred" ]; then
    bash scripts/eval.sh $device $dataset $mode dev_rev-eo
    bash scripts/eval.sh $device $dataset $mode test_rev-eo
fi