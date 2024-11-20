from typing import Callable, Optional

import wandb
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from careamics.lightning import VAEModule

from configs.factory import *
from datasets import create_train_val_datasets
from utils.callbacks import get_callbacks
from utils.io import get_workdir, log_configs


def train_muSplit(
    root_path: str,
    data_path: str,
    param_fn: Callable,
    data_configs_fn: Callable,
    load_data_fn: Callable,
    wandb_project: Optional[str] = None,
) -> None:
    """Train the splitting model.
    
    Parameters
    ----------
    root_path : str
        The root path for the training experiments.
    data_path : str
        The path to the data directory.
    param_fn : Callable
        The function to get the parameters for the specific experiment.
        Select among the ones in `configs/parameters`.
    data_configs_fn : Callable
        The function to get the data configurations for the specific experiment.
        Select among the ones in `configs/data`.
    load_data_fn : Callable
        The function to load the data for the specific experiment.
        Select among the ones in `datasets`.
    wandb_project : Optional[str], optional
        The name of the wandb project, by default None.
    
    Returns
    -------
    None
    
    Examples
    --------
    ```python
    from configs.data.ht_iba1_ki64_2023 import get_data_configs
    from configs.parameters.ht_iba1_ki64_2023 import get_musplit_parameters
    from datasets.ht_iba1_ki64_2023 import get_train_val_data
    
    train_muSplit(
        root_path="/path/to/root",
        data_path="/path/to/data",
        param_fn=get_musplit_parameters,
        data_configs_fn=get_data_configs,
        load_data_fn=get_train_val_data,
        wandb_project="my_project",
    )
    ```
    """
    params = param_fn()
    
    # get datasets and dataloaders
    train_data_config, val_data_config = data_configs_fn()
    train_dset, val_dset, _, data_stats = create_train_val_datasets(
        datapath=data_path,
        train_config=train_data_config,
        val_config=val_data_config,
        test_config=val_data_config, # TODO: check this
        load_data_func=load_data_fn,
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
    dirname = f"{params['algorithm']}_{str(train_data_config.data_type).split('.')[-1]}" 
    logdir, _ = get_workdir(root_path, dirname)
    print(f"Log directory: {logdir}")
    train_data_config.data_path = data_path
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