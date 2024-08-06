CKPT_DIR=./ckpt/release_simi
TABLE_ROOT=./table/release_simi
LABEL_MAP=./table/label.json
MODEL=resnet50
LR=0.001
PATIENT=30
BATCHSIZE=68
EPOCHS=50
MAIN_SCRIPT=train.py

if ! [ -d "$CKPT_DIR" ]; then
    echo "Directory '$CKPT_DIR' does not exist. Creating it now."
    mkdir "$CKPT_DIR"
fi

if ! [ -d "$CKPT_DIR/wce" ]; then
    echo "Directory '$CKPT_DIR/wce' does not exist. Creating it now."
    mkdir "$CKPT_DIR/wce"
fi

CUDA_VISIBLE_DEVICES=4,5,6,7 python $MAIN_SCRIPT \
    --data_table $TABLE_ROOT/realse_0half.json --label_map $LABEL_MAP \
    --batchsize $BATCHSIZE  --lr $LR --epochs $EPOCHS --patient $PATIENT\
    --ckpt_dir $CKPT_DIR/wce \
    --weight_loss \
    --train > $CKPT_DIR/wce/train.log
