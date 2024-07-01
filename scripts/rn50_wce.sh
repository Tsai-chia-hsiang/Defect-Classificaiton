CKPT_DIR=./ckpt/test50
TABLE_ROOT=./dataset/train_valid_test
LABEL_MAP=./dataset/label.json
MODEL=resnet50
LR=0.001
PATIENT=30
BATCHSIZE=68
EPOCHS=50

if ! [ -d "$CKPT_DIR" ]; then
    echo "Directory '$CKPT_DIR' does not exist. Creating it now."
    mkdir "$CKPT_DIR"
fi

if ! [ -d "$CKPT_DIR/baseline" ]; then
    echo "Directory '$CKPT_DIR/baseline' does not exist. Creating it now."
    mkdir "$CKPT_DIR/baseline"
fi

CUDA_VISIBLE_DEVICES=4,5,6,7 python train_test.py \
    --data_table $TABLE_ROOT/baseline.json --label_map $LABEL_MAP \
    --batchsize $BATCHSIZE  --lr $LR --epochs $EPOCHS --patient $PATIENT\
    --ckpt_dir $CKPT_DIR/baseline \
    --train --test > $CKPT_DIR/baseline/train_test.log


if ! [ -d "$CKPT_DIR/baseline_wce" ]; then
    echo "Directory '$CKPT_DIR/baseline_wce' does not exist. Creating it now."
    mkdir "$CKPT_DIR/baseline_wce"
fi

CUDA_VISIBLE_DEVICES=4,5,6,7 python train_test.py \
    --data_table $TABLE_ROOT/baseline.json --label_map $LABEL_MAP \
    --batchsize $BATCHSIZE  --lr $LR --epochs $EPOCHS --patient $PATIENT\
    --ckpt_dir $CKPT_DIR/baseline_wce \
    --weight_loss \
    --train --test > $CKPT_DIR/baseline_wce/train_test.log

if ! [ -d "$CKPT_DIR/aug_wce" ]; then
    echo "Directory '$CKPT_DIR/aug_wce' does not exist. Creating it now."
    mkdir "$CKPT_DIR/aug_wce"
fi

CUDA_VISIBLE_DEVICES=4,5,6,7 python train_test.py \
    --data_table $TABLE_ROOT/aug.json --label_map $LABEL_MAP \
    --batchsize $BATCHSIZE  --lr $LR --epochs $EPOCHS --patient $PATIENT\
    --ckpt_dir $CKPT_DIR/aug_wce \
    --weight_loss \
    --train --test > $CKPT_DIR/aug_wce/train_test.log



if ! [ -d "$CKPT_DIR/down_wce" ]; then
    echo "Directory '$CKPT_DIR/down_wce' does not exist. Creating it now."
    mkdir "$CKPT_DIR/down_wce"
fi

CUDA_VISIBLE_DEVICES=4,5,6,7 python train_test.py \
    --data_table $TABLE_ROOT/down.json --label_map $LABEL_MAP \
    --batchsize $BATCHSIZE  --lr $LR --epochs $EPOCHS --patient $PATIENT\
    --ckpt_dir $CKPT_DIR/down_wce \
    --weight_loss \
    --train --test > $CKPT_DIR/down_wce/train_test.log
