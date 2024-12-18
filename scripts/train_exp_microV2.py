import json
import os
import argparse
import socket
import sys

# TODO: sorry for this hack :(
sys.path.insert(0, "/home/federico.carrara/Documents/projects/microSplit-reproducibility/")
sys.path.insert(0, "/home/igor.zubarev/projects/microSplit-reproducibility/")
sys.path.insert(0, "/home/igor.zubarev/projects/careamics/src")

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional, Union

# import git
import torch
import wandb
from pydantic import BaseModel, ConfigDict
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset

from careamics.config import VAEAlgorithmConfig
from careamics.config.architectures import LVAEModel
from careamics.config.likelihood_model import (
    GaussianLikelihoodConfig,
    NMLikelihoodConfig,
)
from careamics.config.nm_model import GaussianMixtureNMConfig, MultiChannelNMConfig
from careamics.config.optimizer_models import LrSchedulerModel, OptimizerModel
from careamics.lightning import VAEModule
from careamics.lvae_training.train_utils import get_new_model_version
from careamics.models.lvae.noise_models import noise_model_factory

from datasets import load_train_val_exp_microscopy, create_train_val_datasets
from configs.exp_microscopy_v1 import get_data_configs


# --- Custom parameters # TODO move to a separate file
img_size: int = [64, 64]
"""Spatial size of the input image."""
target_channels: int = 2
"""Number of channels in the target image."""
multiscale_count: int = 1
"""The number of LC inputs plus one (the actual input)."""
predict_logvar: Optional[Literal["pixelwise"]] = "pixelwise"
"""Whether to compute also the log-variance as LVAE output."""
loss_type: Optional[Literal["musplit", "denoisplit", "denoisplit_musplit"]] = "musplit"
"""The type of reconstruction loss (i.e., likelihood) to use."""
nm_paths: Optional[tuple[str]] = [
    "/group/jug/ashesh/training_pre_eccv/noise_model/2402/221/GMMNoiseModel_ER-GT_all.mrc__6_4_Clip0.0-1.0_Sig0.125_UpNone_Norm0_bootstrap.npz",
    "/group/jug/ashesh/training_pre_eccv/noise_model/2402/225/GMMNoiseModel_Microtubules-GT_all.mrc__6_4_Clip0.0-1.0_Sig0.125_UpNone_Norm0_bootstrap.npz",
]
"""The paths to the pre-trained noise models for the different channels."""
# TODO: add denoisplit-musplit weights


# --- Training parameters
# TODO: replace with PR #225
class TrainingConfig(BaseModel):
    """Configuration for training a VAE model."""

    model_config = ConfigDict(
        validate_assignment=True, arbitrary_types_allowed=True, extra="allow"
    )

    batch_size: int = 64
    """The batch size for training."""
    precision: int = "16-mixed"
    """The precision to use for training."""
    lr: float = 1e-3
    """The learning rate for training."""
    lr_scheduler_patience: int = 30 // 4 # reduce //4
    """The patience for the learning rate scheduler."""
    earlystop_patience: int = 200 // 4# reduce /4
    """The patience for the learning rate scheduler."""
    max_epochs: int = 400 // 4 # reduce /4
    """The maximum number of epochs to train for."""
    num_workers: int = 4
    """The number of workers to use for data loading."""
    grad_clip_norm_value: int = 0.5
    """The value to use for gradient clipping (see lightning `Trainer`)."""
    gradient_clip_algorithm: int = "value"
    """The algorithm to use for gradient clipping (see lightning `Trainer`)."""



### --- Functions to create datasets and model


# TODO move to a separate file
def create_split_lightning_model(
    algorithm: str,
    loss_type: str,
    img_size: tuple = (64, 64),
    multiscale_count: int = 1,
    predict_logvar: Optional[Literal["pixelwise"]] = None,
    target_ch: int = 1,
    NM_paths: Optional[list[Path]] = None,
    training_config: TrainingConfig = TrainingConfig(),
    data_mean: Optional[torch.Tensor] = None,
    data_std: Optional[torch.Tensor] = None,
) -> VAEModule:
    """Instantiate the muSplit lightining model."""
    lvae_config = LVAEModel(
        architecture="LVAE",
        input_shape=img_size,
        multiscale_count=multiscale_count,
        z_dims=[128, 128, 128, 128],
        output_channels=target_ch,
        predict_logvar=predict_logvar,
        analytical_kl=False,
    )

    # gaussian likelihood
    if loss_type in ["musplit", "denoisplit_musplit"]:
        gaussian_lik_config = GaussianLikelihoodConfig(
            predict_logvar=predict_logvar,
            logvar_lowerbound=-5.0,  # TODO: find a better way to fix this
        )
    else:
        gaussian_lik_config = None
    # noise model likelihood
    if loss_type in ["denoisplit", "denoisplit_musplit"]:
        assert NM_paths is not None, "A path to a pre-trained noise model is required."
        gmm_list = []
        for NM_path in NM_paths:
            gmm_list.append(
                GaussianMixtureNMConfig(
                    model_type="GaussianMixtureNoiseModel",
                    path=NM_path,
                )
            )
        noise_model_config = MultiChannelNMConfig(noise_models=gmm_list)
        nm = noise_model_factory(noise_model_config)
        nm_lik_config = NMLikelihoodConfig(
            noise_model=nm,
            data_mean=data_mean,
            data_std=data_std,
        )
    else:
        noise_model_config = None
        nm_lik_config = None

    opt_config = OptimizerModel(
        name="Adamax",
        parameters={
            "lr": training_config.lr,
            "weight_decay": 0,
        },
    )
    lr_scheduler_config = LrSchedulerModel(
        name="ReduceLROnPlateau",
        parameters={
            "mode": "min",
            "factor": 0.5,
            "patience": training_config.lr_scheduler_patience,
            "verbose": True,
            "min_lr": 1e-12,
        },
    )

    vae_config = VAEAlgorithmConfig(
        algorithm_type="vae",
        algorithm=algorithm,
        loss=loss_type,
        model=lvae_config,
        gaussian_likelihood_model=gaussian_lik_config,
        noise_model=noise_model_config,
        noise_model_likelihood_model=nm_lik_config,
        optimizer=opt_config,
        lr_scheduler=lr_scheduler_config,
    )

    return VAEModule(algorithm_config=vae_config)


# --- Utils
def get_new_model_version(model_dir: Union[Path, str]) -> int:
    """Create a unique version ID for a new model run."""
    versions = []
    for version_dir in os.listdir(model_dir):
        try:
            versions.append(int(version_dir))
        except:
            print(
                f"Invalid subdirectory:{model_dir}/{version_dir}. Only integer versions are allowed"
            )
            exit()
    if len(versions) == 0:
        return "0"
    return f"{max(versions) + 1}"


def get_workdir(
    root_dir: str,
    model_name: str,
) -> tuple[Path, Path]:
    """Get the workdir for the current model.

    It has the following structure: "root_dir/YYMM/model_name/version"
    """
    rel_path = datetime.now().strftime("%y%m")
    cur_workdir = os.path.join(root_dir, rel_path)
    Path(cur_workdir).mkdir(exist_ok=True, parents=True)

    rel_path = os.path.join(rel_path, model_name)
    cur_workdir = os.path.join(root_dir, rel_path)
    Path(cur_workdir).mkdir(exist_ok=True)

    rel_path = os.path.join(rel_path, get_new_model_version(cur_workdir))
    cur_workdir = os.path.join(root_dir, rel_path)
    try:
        Path(cur_workdir).mkdir(exist_ok=False)
    except FileExistsError:
        print(f"Workdir {cur_workdir} already exists.")
    return cur_workdir, rel_path


# def get_git_status() -> dict[Any]:
#     curr_dir = os.path.dirname(os.path.realpath(__file__))
#     repo = git.Repo(curr_dir, search_parent_directories=True)
#     git_config = {}
#     git_config["changedFiles"] = [item.a_path for item in repo.index.diff(None)]
#     git_config["branch"] = repo.active_branch.name
#     git_config["untracked_files"] = repo.untracked_files
#     git_config["latest_commit"] = repo.head.object.hexsha
#     return git_config


def main(rootpath: str, wandb_project: str):

    train_data_config, val_data_config = get_data_configs()
    training_config = TrainingConfig()
    train_dset, val_dset, data_stats = create_train_val_datasets(
    datapath='/localscratch/data/MDCK_MitoDeepRed639_AlphaBetaTub488/',
    train_config=train_data_config,
    val_config=val_data_config,
    load_data_func=load_train_val_exp_microscopy
)
    train_dloader = DataLoader(
        train_dset,
        batch_size=training_config.batch_size,
        num_workers=training_config.num_workers,
        shuffle=True,
    )
    val_dloader = DataLoader(
        val_dset,
        batch_size=training_config.batch_size,
        num_workers=training_config.num_workers,
        shuffle=False,
    )

    algo = "musplit" if loss_type == "musplit" else "denoisplit"
    lightning_model = create_split_lightning_model(
        algorithm=algo,
        loss_type=loss_type,
        img_size=img_size,
        multiscale_count=multiscale_count,
        predict_logvar=predict_logvar,
        target_ch=target_channels,
        NM_paths=nm_paths,
        training_config=training_config,
        data_mean=data_stats[0],
        data_std=data_stats[1],
    )

    lc_tag = "with" if multiscale_count > 1 else "no"
    workdir, exp_tag = get_workdir(rootpath, f"{algo}_{lc_tag}_LC")
    print(f"Current workdir: {workdir}")

    # Define the logger
    # project_name = "_".join(("careamics", algo))
    # if project_name == "_".join(("careamics", algo)):
    #     raise ValueError("Please create your own project name for wandb.")
    if wandb_project != "none":
        custom_logger = WandbLogger(
            name=os.path.join(socket.gethostname(), exp_tag),
            save_dir=workdir,
            project=wandb_project,
        )
    else:
        custom_logger = None

    # Define callbacks (e.g., ModelCheckpoint, EarlyStopping, etc.)
    custom_callbacks = [
        EarlyStopping(
            monitor="val_loss",
            min_delta=1e-6,
            patience=training_config.earlystop_patience,
            mode="min",
            verbose=True,
        ),
        ModelCheckpoint(
            dirpath=workdir,
            filename="best-{epoch}",
            monitor="val_loss",
            save_top_k=1,
            save_last=True,
            mode="min",
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # Save configs and git status (for debugging)
    algo_config = lightning_model.algorithm_config
    data_config = train_data_config
    # temp -> remove fields that we don't want to save
    loss_config = deepcopy(asdict(lightning_model.loss_parameters))
    del loss_config["noise_model_likelihood"]
    del loss_config["gaussian_likelihood"]

    # with open(os.path.join(workdir, "git_config.json"), "w") as f:
    #     json.dump(get_git_status(), f, indent=4)

    with open(os.path.join(workdir, "algorithm_config.json"), "w") as f:
        f.write(algo_config.model_dump_json(indent=4))

    with open(os.path.join(workdir, "training_config.json"), "w") as f:
        f.write(training_config.model_dump_json(indent=4))

    with open(os.path.join(workdir, "data_config.json"), "w") as f:
        f.write(data_config.model_dump_json(indent=4))

    with open(os.path.join(workdir, "loss_config.json"), "w") as f:
        json.dump(loss_config, f, indent=4)

    # Save Configs in WANDB
    if custom_logger is not None:
        custom_logger.experiment.config.update({"algorithm": algo_config.model_dump()})

        custom_logger.experiment.config.update({"training": training_config.model_dump()})

        custom_logger.experiment.config.update({"data": data_config.model_dump()})

        custom_logger.experiment.config.update({"loss_params": loss_config})

    # Train the model
    trainer = Trainer(
        max_epochs=training_config.max_epochs,
        accelerator="gpu",
        enable_progress_bar=True,
        logger=custom_logger,
        callbacks=custom_callbacks,
        precision=training_config.precision,
        gradient_clip_val=training_config.grad_clip_norm_value,  # only works with `accelerator="gpu"`
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
        "--rootpath",
        type=str,
        help="The root path for the training experiments.",
        required=True,
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        help="The name of the wandb project.",
        required=True,
    )
    args = parser.parse_args()
    main(args.rootpath, args.wandb_project)
