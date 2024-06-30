CKPT_DIR=./ckpt/test
TABLE_ROOT=./dataset/train_valid_test
LABEL_MAP=./dataset/label.json
MODEL=resnet50

CUDA_VISIBLE_DEVICES=0,1,2,3  python train_test.py \
    --data_table $TABLE_ROOT/aug.json --label_map $LABEL_MAP\
    --batchsize 60  --lr 0.001 --epochs 80 \
    --ckpt_dir $CKPT_DIR/aug_wce --weight_loss \
    --train --test > $CKPT_DIR/aug_wce/train_test.log

CUDA_VISIBLE_DEVICES=0,1,2,3 python train_test.py \
    --data_table $TABLE_ROOT/baseline.json --label_map $LABEL_MAP \
    --batchsize 60  --lr 0.001 --epochs 80 \
    --ckpt_dir $CKPT_DIR/baseline_focal \
    --weight_loss --loss focal \
    --train --test > $CKPT_DIR/baseline_focal/train_test.log

CUDA_VISIBLE_DEVICES=0,1,2,3 python train_test.py \
    --data_table $TABLE_ROOT/down.json --label_map $LABEL_MAP\
    --batchsize 60  --lr 0.001 --epochs 80 \
    --ckpt_dir $CKPT_DIR/down_focal \
    --weight_loss --loss focal \
    --train --test > $CKPT_DIR/down_focal/train_test.log

CUDA_VISIBLE_DEVICES=0,1,2,3 python train_test.py \
    --data_table $TABLE_ROOT/aug.json --label_map $LABEL_MAP\
    --batchsize 60  --lr 0.001 --epochs 80 \
    --ckpt_dir $CKPT_DIR/aug_focal \
    --weight_loss --loss focal \
    --train --test > $CKPT_DIR/aug_focal/train_test.log