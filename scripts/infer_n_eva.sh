CKPT_DIR=./ckpt/test/aug_wce

CUDA_VISIBLE_DEVICES=7 python inference.py --model_root $CKPT_DIR
python eva.py --ans ./dataset/origin_ans.csv --predict $CKPT_DIR/submit.csv > $CKPT_DIR/test.log