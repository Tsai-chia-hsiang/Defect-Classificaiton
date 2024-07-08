CKPT_DIR=./ckpt/release/patch_independent
TABLE_ROOT=./table/patches
LABEL_MAP=./table/label.json 
MODEL=resnet34
LR=0.0005
PATIENT=30
BATCHSIZE=256
EPOCHS=50
MAIN_SCRIPT=train_test.py

if ! [ -d "$CKPT_DIR" ]; then
    echo "Directory '$CKPT_DIR' does not exist. Creating it now."
    mkdir "$CKPT_DIR"
fi
WD=$CKPT_DIR/channel
if ! [ -d $WD ]; then
    echo "Directory '$WD' does not exist. Creating it now."
    mkdir "$WD"
fi

CUDA_VISIBLE_DEVICES=0,1,2,3 python $MAIN_SCRIPT \
    --data_table $TABLE_ROOT/patches.json --label_map $LABEL_MAP \
    --dtype patch --patch_independent \
    --coor channel --using_model $MODEL \
    --batchsize $BATCHSIZE  --lr $LR --epochs $EPOCHS --patient $PATIENT\
    --ckpt_dir $WD \
    --train --test > $WD/train_test.log

python texture_classify.py --predict_csv $WD/test_pred.csv > $WD/full_test.log