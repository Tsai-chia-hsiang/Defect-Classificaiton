CKPT_DIR=./ckpt/test/
TABLE_ROOT=./dataset/train_valid_test
LABEL_MAP=./dataset/label.json
MODEL=resnet50

CUDA_VISIBLE_DEVICES=4,5,6,7 python train_test.py \
    --data_table $TABLE_ROOT/baseline.json --label_map $LABEL_MAP \
    --batchsize 60  --lr 0.001 --epochs 80 \
    --ckpt_dir $CKPT_DIR/baseline \
    --train --test > $CKPT_DIR/baseline/train_test.log

CUDA_VISIBLE_DEVICES=4,5,6,7 python train_test.py \
    --data_table $TABLE_ROOT/down.json --label_map $LABEL_MAP\
    --batchsize 60  --lr 0.001 --epochs 80 \
    --ckpt_dir $CKPT_DIR/down \
    --train --test > $CKPT_DIR/down/train_test.log

CUDA_VISIBLE_DEVICES=4,5,6,7 python train_test.py \
    --data_table $TABLE_ROOT/aug.json --label_map $LABEL_MAP\
    --batchsize 60  --lr 0.001 --epochs 80 \
    --ckpt_dir $CKPT_DIR/aug \
    --train --test > $CKPT_DIR/aug/train_test.log

CUDA_VISIBLE_DEVICES=4,5,6,7 python train_test.py \
    --data_table $TABLE_ROOT/baseline.json --label_map $LABEL_MAP \
    --batchsize 60  --lr 0.001 --epochs 80 --weight_loss \
    --ckpt_dir $CKPT_DIR/baseline_wce \
    --train --test > $CKPT_DIR/baseline_wce/train_test.log

CUDA_VISIBLE_DEVICES=4,5,6,7 python train_test.py \
    --data_table $TABLE_ROOT/down.json --label_map $LABEL_MAP\
    --batchsize 60  --lr 0.001 --epochs 80 --weight_loss \
    --ckpt_dir $CKPT_DIR/down_wce \
    --train --test > $CKPT_DIR/down_wce/train_test.log

