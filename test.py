from pathlib import Path
from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader
import pandas as pd
from libs.model import ResNet_CLS_Contrastive_Model as resnext_model
from libs.dataset import FullImg_Dataset
from libs.cfg import load_config
from libs.trainer import Supervised_CLS_Contrastive_Learning_Wrapper
from libs import read_json, set_seed , write_json 
from libs.log import get_logger
def main(config, args):

    logger = get_logger(name=__name__, file=Path(args.trained_weights).parent/"test.log")

    test_set = FullImg_Dataset.build_datasets(
        file_table = read_json(config.dataset.test_table), 
        label_map = read_json(config.dataset.label_map),
        dsize=config.dataset.input_size, logger=logger
    )['test']

    model = resnext_model.get_model(
        ncls=test_set.ncls, grayscale=True, depth=str(config.model.depth), 
        pretrained=args.trained_weights, logger=logger
    )
    infer_warpper = Supervised_CLS_Contrastive_Learning_Wrapper(
        model=model, device= torch.device(config.device),
        dp=config.model.dp , logger=logger
    ) 
    eval_result = infer_warpper.inference_one_epoch(
        inference_loader=DataLoader(test_set, batch_size=config.dataset.val_batch),
        return_pred=False, confusion_matrix='pd', metrics_tolist=True
    )
    write_json({k:v for k,v in eval_result.items() if k != "confusion matrix"}, Path(args.trained_weights).parent/f"eva.json")
    eval_result['confusion matrix'].to_csv(
        Path(args.trained_weights).parent/f"cm.csv",
        index=False
    )
    logger.info(f"macro F1 : {eval_result['macro f1']}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs")/"default.yaml")
    parser.add_argument("--trained_weights", type=Path)
    args = parser.parse_args()
    config = load_config(yaml_file=args.config)    
    set_seed(config.seed)
    main(config=config, args=args)
