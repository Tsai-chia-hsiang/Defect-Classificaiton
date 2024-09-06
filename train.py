from pathlib import Path
from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from libs.model import ResNet_CLS_Contrastive_Model as resnet_model
from libs.dataset import FullImg_Dataset
from libs.cfg import load_config
from libs.trainer import Supervised_CLS_Contrastive_Learning_Wrapper
from libs.log import get_logger, remove_old_tf_evenfile
from libs import read_json, set_seed  


def main(config):
    ckpt = Path(config.model.ckpt)
    remove_old_tf_evenfile(ckpt)
    ckpt.mkdir(parents=True, exist_ok=True)

    logger = get_logger(name=__name__, file=ckpt/"training.log")

    logger.info(config)
   
    datasets:dict[str, FullImg_Dataset] = FullImg_Dataset.build_datasets(
        file_table = read_json(config.dataset.table), 
        label_map = read_json(config.dataset.label_map),
        logger=logger,
        dsize=config.dataset.input_size
    )
    
    model = resnet_model.get_model(
        ncls=datasets['train'].ncls, grayscale=True, 
        depth=str(config.model.depth), logger=logger, 
        pretrained=config.model.pretrained
    )

    trainer_warpper = Supervised_CLS_Contrastive_Learning_Wrapper(
        model=model, device= torch.device(config.device),
        dp=config.model.dp, logger=logger, 
        board=SummaryWriter(config.model.ckpt)
    ) 
    trainer_warpper.before_train_setting(
        ckpt_dir=ckpt, 
        model_name=str(model),
        batch={
            'train':config.dataset.train_batch, 
            'valid':config.dataset.val_batch,
        },
        train_set=datasets['train'], val_set=datasets['valid'],
        g_seed=config.seed
    )
    if config.val_only:
        result = trainer_warpper.inference_one_epoch(
            inference_loader=DataLoader(dataset=datasets['valid'], batch_size=config.dataset.val_batch, pin_memory=True),
            return_pred=False, confusion_matrix='pd'
        )
        return result
    
    best_weights_path = trainer_warpper.train_model(**config.opt)
    return best_weights_path

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs")/"default.yaml")
    args = parser.parse_args()
    config = load_config(yaml_file=args.config)    
    set_seed(config.seed)
    ret = main(config=config)
    print(ret)