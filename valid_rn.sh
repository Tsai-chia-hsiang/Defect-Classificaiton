CKPT_DIR=./ckpt/rn_50_w_down
DATA_TABLE=./dataset/5_folds.json
LABEL_MAP=./dataset/label.json
MODEL=resnet50

python train_test.py \
    --data_table $DATA_TABLE --label_map $LABEL_MAP\
    --batchsize 100  --ckpt_dir $CKPT_DIR\
    --valid_only --device 0