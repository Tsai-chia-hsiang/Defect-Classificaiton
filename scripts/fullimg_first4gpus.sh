CKPT_DIR=./ckpt/test
TABLE_ROOT=./dataset/train_valid_test
LABEL_MAP=./dataset/label.json
MODEL=resnet50
LR=0.001
PATIENT=30
BATCHSIZE=68
EPOCHS=50
MAIN_SCRIPT=train_test.py

if ! [ -d "$CKPT_DIR" ]; then
    echo "Directory '$CKPT_DIR' does not exist. Creating it now."
    mkdir "$CKPT_DIR"
fi


if ! [ -d "$CKPT_DIR/aug" ]; then
    echo "Directory '$CKPT_DIR/aug' does not exist. Creating it now."
    mkdir "$CKPT_DIR/aug"
fi

CUDA_VISIBLE_DEVICES=0,1,2,3 python $MAIN_SCRIPT \
    --data_table $TABLE_ROOT/aug.json --label_map $LABEL_MAP \
    --batchsize $BATCHSIZE  --lr $LR --epochs $EPOCHS --patient $PATIENT\
    --ckpt_dir $CKPT_DIR/aug \
    --train --test > $CKPT_DIR/aug/train_test.log

if ! [ -d "$CKPT_DIR/baseline_focal" ]; then
    echo "Directory '$CKPT_DIR/baseline_focal' does not exist. Creating it now."
    mkdir "$CKPT_DIR/baseline_focal"
fi

CUDA_VISIBLE_DEVICES=0,1,2,3 python $MAIN_SCRIPT \
    --data_table $TABLE_ROOT/baseline.json --label_map $LABEL_MAP \
    --batchsize $BATCHSIZE  --lr $LR --epochs $EPOCHS --patient $PATIENT\
    --ckpt_dir $CKPT_DIR/baseline_focal \
    --weight_loss --loss focal\
    --train --test > $CKPT_DIR/baseline_focal/train_test.log

if ! [ -d "$CKPT_DIR/aug_focal" ]; then
    echo "Directory '$CKPT_DIR/aug_focal' does not exist. Creating it now."
    mkdir "$CKPT_DIR/aug_focal"
fi

CUDA_VISIBLE_DEVICES=0,1,2,3 python $MAIN_SCRIPT \
    --data_table $TABLE_ROOT/aug.json --label_map $LABEL_MAP \
    --batchsize $BATCHSIZE  --lr $LR --epochs $EPOCHS --patient $PATIENT\
    --ckpt_dir $CKPT_DIR/aug_focal \
    --weight_loss --loss focal\
    --train --test > $CKPT_DIR/aug_focal/train_test.log


if ! [ -d "$CKPT_DIR/down" ]; then
    echo "Directory '$CKPT_DIR/down' does not exist. Creating it now."
    mkdir "$CKPT_DIR/down"
fi

CUDA_VISIBLE_DEVICES=0,1,2,3 python $MAIN_SCRIPT \
    --data_table $TABLE_ROOT/down.json --label_map $LABEL_MAP \
    --batchsize $BATCHSIZE  --lr $LR --epochs $EPOCHS --patient $PATIENT\
    --ckpt_dir $CKPT_DIR/down \
    --train --test > $CKPT_DIR/down/train_test.log

if ! [ -d "$CKPT_DIR/down_focal" ]; then
    echo "Directory '$CKPT_DIR/down_focal' does not exist. Creating it now."
    mkdir "$CKPT_DIR/down_focal"
fi

CUDA_VISIBLE_DEVICES=0,1,2,3 python $MAIN_SCRIPT \
    --data_table $TABLE_ROOT/down.json --label_map $LABEL_MAP \
    --batchsize $BATCHSIZE  --lr $LR --epochs $EPOCHS --patient $PATIENT\
    --ckpt_dir $CKPT_DIR/down_focal \
    --weight_loss --loss focal\
    --train --test > $CKPT_DIR/down_focal/train_test.log