from yacs.config import CfgNode as CN

_C = CN()
_C.seed = 891122
_C.device = "cuda:0"
_C.val_only=False
_C.inference_only=False
# model
_C.model = CN()
_C.model.depth = "rx50_32"
_C.model.pretrained = ""
_C.model.dp = False
_C.model.ckpt = ""

#opt
_C.opt = CN()
_C.opt.lr = 0.01
_C.opt.epochs = 30
_C.opt.optr= "adamw"
_C.opt.val_epochs = 1
_C.opt.warm_up = 0
_C.opt.cls_weight = 2
_C.opt.cl_weight = 0.6
_C.opt.debug = -1

# dataset
_C.dataset = CN()
_C.dataset.table = ""
_C.dataset.test_table = ""
_C.dataset.label_map = ""
_C.dataset.train_batch = 28
_C.dataset.val_batch = 28
_C.dataset.input_size = None