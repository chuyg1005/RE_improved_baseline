DEVICE=$1;
DATASET=$2;

#for lamb in 0.2 0.4 0.6 0.8 1.0; do
#  export LAMBDA=$lamb;
#  bash scripts/train.sh $DEVICE $DATASET MixDebias;
#done
export LAMBDA=0.5;
bash scripts/train.sh $DEVICE $DATASET MixDebias;
