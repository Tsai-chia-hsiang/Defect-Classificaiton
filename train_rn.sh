CKPT_DIR=./ckpt/rn_50/
DATA_TABLE=./dataset/5_folds.json
LABEL_MAP=./dataset/label.json
MODEL=resnet50

CUDA_VISIBLE_DEVICES=4,5,6,7 python train_test.py \
    --data_table $DATA_TABLE --label_map $LABEL_MAP\
    --batchsize 60  --lr 0.001 --epochs 50\
    --ckpt_dir $CKPT_DIR\
    --no_valid