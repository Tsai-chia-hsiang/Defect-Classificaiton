CKPT_DIR=./ckpt/release
TABLE_ROOT=./dataset/release
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

DSTDIR=$CKPT_DIR/aug_epoch50
if ! [ -d $DSTDIR ]; then
    echo "Directory $DSTDIR does not exist. Creating it now."
    mkdir $DSTDIR
fi

CUDA_VISIBLE_DEVICES=0,1,2,3 python $MAIN_SCRIPT \
    --data_table $TABLE_ROOT/release_aug_t0half.json --label_map $LABEL_MAP \
    --batchsize $BATCHSIZE  --lr $LR --epochs $EPOCHS --patient $PATIENT\
    --weight_loss \
    --ckpt_dir $DSTDIR \
    --train --test > $DSTDIR/train_test.log
