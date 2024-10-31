import argparse
import os
import sys
from pathlib import Path
from typing import Union

import torch
from torch.utils.data import DataLoader

# TODO: sorry for this :(
sys.path.insert(0, "/home/federico.carrara/Documents/projects/microSplit-reproducibility/")

from careamics.lightning import VAEModule

from configs import get_algorithm_config
from data.data_utils import GridAlignement, load_tiff
from datasets import load_train_val_sox2golgi_v2, create_train_val_datasets
from configs.sox2golgi_v2 import get_data_configs
from utils import fix_seeds
from utils.io import get_model_checkpoint, load_config


# def load_configs(config_dir: Union[str, Path]) -> tuple:
#     algo_config = load_config(config_dir, "algorithm")
#     training_config = load_config(config_dir, "training")
#     data_config = load_config(config_dir, "data")
#     return algo_config, training_config, data_config


def load_checkpoint(ckpt_dir: Union[str, Path], best: bool = True) -> dict:
    """Load the checkpoint from the given directory."""
    if os.path.isdir(ckpt_dir):
        ckpt_fpath = get_model_checkpoint(ckpt_dir, mode="best" if best else "last")
    else:
        assert os.path.isfile(ckpt_dir)
        ckpt_fpath = ckpt_dir

    ckpt = torch.load(ckpt_fpath)
    print(f"Loading checkpoint from: '{ckpt_fpath}' - Epoch: {ckpt["epoch"]}")
    return ckpt

    
    
def evaluate():
    params = get_musplit_parameters()
    
    # get datasets and dataloaders
    train_dset, val_dset, _, data_stats = create_train_val_datasets(
        datapath=data_path,
        train_config=train_data_config,
        val_config=val_data_config,
        test_config=val_data_config, # TODO: check this
        load_data_func=get_train_val_data,
    )
    train_dloader = DataLoader(
        train_dset,
        batch_size=params["batch_size"],
        num_workers=params["num_workers"],
        shuffle=True,
    )
    val_dloader = DataLoader(
        val_dset,
        batch_size=params["batch_size"],
        num_workers=params["num_workers"],
        shuffle=False,
    )
    
    model_config = load_config(ckpt_path, "model")
    # TODO: all the previous configs can also be istantiated from a single function...
    algo_config = get_algorithm_config(
        algorithm=params["algorithm"],
        model_config=model_config,
    )
    
    # load checkpoint
    checkpoint = load_checkpoint(ckpt_path, best=True)
    
    # init lightning model
    lightning_model = VAEModule(algorithm_config=algo_config)
    lightning_model.load_state_dict(checkpoint['state_dict'], strict=True)
    lightning_model.eval()
    lightning_model.cuda()
    
    # get predictions
    pred_tiled = get_dset_predictions(
        model=lightning_model,
        dset=test_dset,
        batch_size=batch_size,
        num_workers=num_workers,
        mmse_count=mmse_count,
        loss_type=algo_config["loss"],
    )
    
    # stich predictions
    ...

    # ignore pixels
    ...
    
    # compute metrics
    ...
    
    # write metrics on a JSON
    # NOTE: JSON file will be updated at each evaluation
    # - we will keep a JSON file for each experiment
    # - each experiment will be tracked by a timestamp
    # - the JSON file will tracked with git
    ...
    

if __name__ == "__main__":
    fix_seeds(0)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        type=str,
        help="The path to the training checkpoints.",
        required=True,
    )
    parser.add_argument(
        "--root_path",
        type=str,
        help="The root path for the training experiments.",
        required=True,
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="The path to the data directory.",
        required=True,
    )
    args = parser.parse_args()
    evaluate()

