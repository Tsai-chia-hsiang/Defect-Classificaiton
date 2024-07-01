CKPT_DIR=./ckpt/patch
TABLE_ROOT=./connectedcomponents/patches
LABEL_MAP=./dataset/label.json 
MODEL=resnet34
LR=0.0005
PATIENT=20
BATCHSIZE=256
EPOCHS=30
MAIN_SCRIPT=train_test.py

if ! [ -d "$CKPT_DIR" ]; then
    echo "Directory '$CKPT_DIR' does not exist. Creating it now."
    mkdir "$CKPT_DIR"
fi
if ! [ -d "$CKPT_DIR/baseline34" ]; then
    echo "Directory '$CKPT_DIR/baseline50' does not exist. Creating it now."
    mkdir "$CKPT_DIR/baseline34"
fi

CUDA_VISIBLE_DEVICES=0,1,2,3 python $MAIN_SCRIPT \
    --data_table $TABLE_ROOT/f_table.json --label_map $LABEL_MAP \
    --patch --using_model $MODEL \
    --batchsize $BATCHSIZE  --lr $LR --epochs $EPOCHS --patient $PATIENT\
    --ckpt_dir $CKPT_DIR/baseline34 \
    --train --test > $CKPT_DIR/baseline34/train_test.log

python texture_classify.py > $CKPT_DIR/baseline34/final.log