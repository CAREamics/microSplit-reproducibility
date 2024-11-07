
import argparse
import sys
from typing import Optional

import wandb
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

# TODO: sorry for this hack :(
sys.path.insert(0, "/home/federico.carrara/Documents/projects/microSplit-reproducibility/")
# sys.path.insert(0, "/home/federico.carrara/Documents/projects/careamics/src")

from careamics.lightning import VAEModule

from configs.factory import *
from datasets import create_train_val_datasets
from utils.callbacks import get_callbacks
from utils.io import get_workdir, log_configs
# NOTE: the following imports are datasets and algorithm dependent
from configs.data.ht_iba1_ki64 import get_data_configs
from configs.parameters.ht_iba1_ki64 import get_musplit_parameters
from datasets.ht_iba1_ki64 import get_train_val_data


# TODO: this whole function is common, so it can also be moved somewhere else
def train_musplit(root_path: str, data_path: str, wandb_project: Optional[str] = None) -> None:
    """Train the splitting model.
    
    Parameters
    ----------
    root_path : str
        The root path for the training experiments.
    data_path : str
        The path to the data directory.
    wandb_project : Optional[str], optional
        The name of the wandb project, by default None.
    
    Returns
    -------
    None
    """
    
    params = get_musplit_parameters()
    
    # get datasets and dataloaders
    train_data_config, val_data_config = get_data_configs()
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
    
    # get configs
    params["data_stats"] = data_stats
    loss_config = get_loss_config(**params)
    model_config = get_model_config(**params)
    gaussian_lik_config, noise_model_config, nm_lik_config = get_likelihood_config(**params)
    training_config = get_training_config(**params)
    lr_scheduler_config = get_lr_scheduler_config(**params)
    optimizer_config = get_optimizer_config(**params)
    # TODO: all the previous configs can also be istantiated from a single function...
    algo_config = get_algorithm_config(
        algorithm=params["algorithm"],
        loss_config=loss_config,
        model_config=model_config,
        gaussian_lik_config=gaussian_lik_config,
        nm_config=noise_model_config,
        nm_lik_config=nm_lik_config,
        lr_scheduler_config=lr_scheduler_config,
        optimizer_config=optimizer_config
    )
    
    # log configs
    dirname = f"{params['algorithm']}_{train_data_config.data_type.split('.')[-1]}" 
    logdir, _ = get_workdir(root_path, dirname)
    print(f"Log directory: {logdir}")
    train_data_config["data_path"] = data_path
    custom_logger = log_configs(
        configs=[algo_config, training_config, train_data_config, loss_config],
        names=["algorithm", "training", "data", "loss"],
        log_dir=logdir,
        wandb_project=wandb_project,
    )
    # init lightning model
    lightning_model = VAEModule(algorithm_config=algo_config)
    
    # train the model
    custom_callbacks = get_callbacks(logdir)
    trainer = Trainer(
        max_epochs=training_config.num_epochs,
        accelerator="gpu",
        enable_progress_bar=True,
        logger=custom_logger,
        callbacks=custom_callbacks,
        precision=training_config.precision,
        gradient_clip_val=training_config.gradient_clip_val,
        gradient_clip_algorithm=training_config.gradient_clip_algorithm,
    )
    trainer.fit(
        model=lightning_model,
        train_dataloaders=train_dloader,
        val_dataloaders=val_dloader,
    )
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    parser.add_argument(
        "--wandb_project",
        type=str,
        help="The name of the wandb project.",
        required=False,
        default=None,
    )
    args = parser.parse_args()
    train_musplit(
        root_path=args.root_path,
        data_path=args.data_path, 
        wandb_project=args.wandb_project
    )