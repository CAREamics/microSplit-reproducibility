
import argparse
import sys
from typing import Optional

import wandb
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

# TODO: sorry for this hack :(
sys.path.insert(0, "/home/federico.carrara/Documents/projects/microSplit-reproducibility/")
sys.path.insert(0, "/home/federico.carrara/Documents/projects/careamics/src")

from careamics.lightning import VAEModule

from configs.factory import *
from utils.callbacks import get_callbacks
from utils.io import get_workdir, log_configs
# NOTE: the following imports are datasets and algorithm dependent
from configs.data.biosr import get_data_configs
from configs.parameters.biosr import get_musplit_parameters
from datasets import load_train_val_sox2golgi_v2, create_train_val_datasets


def train(root_path: str, data_path: str, wandb_project: Optional[str] = None) -> None:
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
    train_data_config, val_data_config, test_data_config = get_data_configs()
    train_dset, val_dset, _, data_stats = create_train_val_datasets(
        datapath=data_path,
        train_config=train_data_config,
        val_config=val_data_config,
        test_config=test_data_config,
        load_data_func=load_train_val_sox2golgi_v2
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
        noise_model_config=noise_model_config,
        nm_lik_config=nm_lik_config,
        training_config=training_config,
        lr_scheduler_config=lr_scheduler_config,
        optimizer_config=optimizer_config
    )
    
    # log configs
    logdir, _ = get_workdir(root_path)
    print(f"Log directory: {logdir}")
    custom_logger = log_configs(
        configs=[algo_config, training_config, train_data_config, loss_config],
        names=["algorithm", "training", "data", "loss"],
        log_dir=logdir,
        wandb_project=wandb_project,
    )
    
    # init lightning model
    lightning_model = VAEModule(**algo_config.model_dump())
    
    # train the model
    custom_callbacks = get_callbacks(logdir)
    trainer = Trainer(
        max_epochs=training_config.max_epochs,
        accelerator="gpu",
        enable_progress_bar=True,
        logger=custom_logger,
        callbacks=custom_callbacks,
        precision=training_config.precision,
        gradient_clip_val=training_config.grad_clip_norm_value,
        gradient_clip_algorithm=training_config.gradient_clip_algorithm,
        limit_train_batches=training_config.limit_train_batches,
        limit_val_batches=training_config.limit_train_batches
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
        required=True,
    )
    args = parser.parse_args()
    train(
        root_path=args.root_path,
        data_path=args.data_path, 
        wandb_project=args.wandb_project
    )
