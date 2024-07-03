CKPT_DIR=./ckpt/patch_test
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

CUDA_VISIBLE_DEVICES=4,5,6,7 python $MAIN_SCRIPT \
    --data_table $TABLE_ROOT/f_table.json --label_map $LABEL_MAP \
    --dtype patchimg --using_model $MODEL \
    --batchsize $BATCHSIZE  --lr $LR --epochs $EPOCHS --patient $PATIENT\
    --ckpt_dir $CKPT_DIR/baseline34 \
    --train --test > $CKPT_DIR/baseline34/train_test.log

python texture_classify.py > $CKPT_DIR/baseline34/final.log

if ! [ -d "$CKPT_DIR/baseline_coo34" ]; then
        echo "Directory '$CKPT_DIR/baseline_coo34' does not exist. Creating it now."
        mkdir "$CKPT_DIR/baseline_coo34"
fi

CUDA_VISIBLE_DEVICES=4,5,6,7 python $MAIN_SCRIPT \
    --data_table $TABLE_ROOT/patches_coo.json --label_map $LABEL_MAP \
    --dtype patchimg --using_model $MODEL \
    --batchsize $BATCHSIZE  --lr $LR --epochs $EPOCHS --patient $PATIENT\
    --ckpt_dir $CKPT_DIR/baseline_coo34 \
    --train --test > $CKPT_DIR/baseline_coo34/train_test.log
python texture_classify.py --predict_csv  $CKPT_DIR/baseline_coo34/test_pred.csv > $CKPT_DIR/baseline_coo34/final.log

if ! [ -d "$CKPT_DIR/baseline_coo34_wce" ]; then
    echo "Directory '$CKPT_DIR/baseline_coo34_wce' does not exist. Creating it now."
    mkdir "$CKPT_DIR/baseline_coo34_wce"
fi

CUDA_VISIBLE_DEVICES=4,5,6,7 python $MAIN_SCRIPT \
    --data_table $TABLE_ROOT/patches_coo.json --label_map $LABEL_MAP \
    --dtype patchimg --using_model $MODEL \
    --batchsize $BATCHSIZE  --lr $LR --epochs $EPOCHS --patient $PATIENT\
    --weight_loss --ckpt_dir $CKPT_DIR/baseline_coo34_wce \
    --train --test > $CKPT_DIR/baseline_coo34_wce/train_test.log

python texture_classify.py --predict_csv  $CKPT_DIR/baseline_coo34_wce/test_pred.csv > $CKPT_DIR/baseline_coo34_wce/final.log

if ! [ -d "$CKPT_DIR/baseline_coo34_focal" ]; then
        echo "Directory '$CKPT_DIR/baseline_coo34_focal' does not exist. Creating it now."
        mkdir "$CKPT_DIR/baseline_coo34_focal"
fi

CUDA_VISIBLE_DEVICES=4,5,6,7 python $MAIN_SCRIPT \
    --data_table $TABLE_ROOT/patches_coo.json --label_map $LABEL_MAP \
    --dtype patchimg --using_model $MODEL \
    --batchsize $BATCHSIZE  --lr $LR --epochs $EPOCHS --patient $PATIENT\
    --weight_loss --loss focal --ckpt_dir $CKPT_DIR/baseline_coo34_focal \
    --train --test > $CKPT_DIR/baseline_coo34_focal/train_test.log
python texture_classify.py --predict_csv  $CKPT_DIR/baseline_coo34_focal/test_pred.csv > $CKPT_DIR/baseline_coo34_focal/final.log
