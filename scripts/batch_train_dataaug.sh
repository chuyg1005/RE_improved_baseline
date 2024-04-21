DEVICE=$1;
DATASET=$2;

# Train
#bash scripts/train.sh $DEVICE $DATASET default;
#bash scripts/train.sh $DEVICE $DATASET EntityMask;
#bash scripts/train.sh $DEVICE $DATASET DataAug;
#bash scripts/train.sh $DEVICE $DATASET RDrop;
#bash scripts/train.sh $DEVICE $DATASET Focal;
#bash scripts/train.sh $DEVICE $DATASET DFocal;
#bash scripts/train.sh $DEVICE $DATASET DFocalAnneal;
#bash scripts/train.sh $DEVICE $DATASET PoE;
#bash scripts/train.sh $DEVICE $DATASET PoEAnneal;
#bash scripts/train.sh $DEVICE $DATASET MixDebias@0.2@0.1;
#bash scripts/train.sh $DEVICE $DATASET MixDebias@0.4@0.1;
#bash scripts/train.sh $DEVICE $DATASET MixDebias@0.6@0.1;
#bash scripts/train.sh $DEVICE $DATASET MixDebias@0.8@0.1;
#bash scripts/train.sh $DEVICE $DATASET MixDebias@1.@0.1;
bash scripts/train.sh $DEVICE $DATASET RDataAug@0.;
bash scripts/train.sh $DEVICE $DATASET RDataAug@0.2;
bash scripts/train.sh $DEVICE $DATASET RDataAug@0.4;
bash scripts/train.sh $DEVICE $DATASET RDataAug@0.6;
bash scripts/train.sh $DEVICE $DATASET RDataAug@0.8;
bash scripts/train.sh $DEVICE $DATASET RDataAug@1.;
