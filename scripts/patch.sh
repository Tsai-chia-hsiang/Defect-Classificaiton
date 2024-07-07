CKPT_DIR=./ckpt/patch
TABLE_ROOT=./table/patches
LABEL_MAP=./table/label.json 
MODEL=resnet34
LR=0.001
PATIENT=30
BATCHSIZE=100
EPOCHS=50
MAIN_SCRIPT=train_test.py

if ! [ -d "$CKPT_DIR" ]; then
    echo "Directory '$CKPT_DIR' does not exist. Creating it now."
    mkdir "$CKPT_DIR"
fi
if ! [ -d "$CKPT_DIR/baseline34" ]; then
    echo "Directory '$CKPT_DIR/baseline34' does not exist. Creating it now."
    mkdir "$CKPT_DIR/baseline34"
fi

CUDA_VISIBLE_DEVICES=4,5,6,7 python $MAIN_SCRIPT \
    --data_table $TABLE_ROOT/patches_src.json --label_map $LABEL_MAP \
    --dtype patchimg --using_model $MODEL \
    --batchsize $BATCHSIZE  --lr $LR --epochs $EPOCHS --patient $PATIENT\
    --weight_loss --ckpt_dir $CKPT_DIR/baseline34 \
    --train --test > $CKPT_DIR/baseline34/train_test.log
