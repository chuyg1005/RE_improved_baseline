# 数据集列表，根据实际情况进行修改
datasets=("tacred" "tacrev" "retacred")
splits=("train" "dev" "test")
modes=("co" "co-o" "eo" "eo-t" "eo-m")

for dataset in "${datasets[@]}"; do
    for split in "${splits[@]}"; do
        for mode in "${modes[@]}"; do
            echo "dataset: $dataset, split: $split, mode: $mode"
            python gen_dataset.py --dataset $dataset --split $split --mode $mode
            echo "处理完成"
        done
    done
done