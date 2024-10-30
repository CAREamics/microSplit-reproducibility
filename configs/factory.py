from typing import Literal

from careamics.config import VAEAlgorithmConfig
from careamics.config.architectures import LVAEModel
from careamics.config.loss_model import LVAELossConfig, KLLossConfig
from careamics.config.nm_model import GaussianMixtureNMConfig, MultiChannelNMConfig
from careamics.config.likelihood_model import GaussianLikelihoodConfig, NMLikelihoodConfig
from careamics.config.training_model import TrainingConfig
from careamics.config.optimizer_models import OptimizerModel, LrSchedulerModel


def get_model_config(**kwargs) -> LVAEModel:
    """Get the model configuration for the muSplit model.
    
    Parameters
    ----------
    ...

    Returns
    -------
    LVAEModel
        The LVAE model configuration.
    """
    return LVAEModel(
        architecture="LVAE",
        input_shape=kwargs["img_size"],
        multiscale_count=kwargs["multiscale_count"],
        z_dims=[128, 128, 128, 128],
        output_channels=kwargs["target_ch"],
        predict_logvar=kwargs["predict_logvar"],
        analytical_kl=False,
    )
    

def get_likelihood_config(
    **kwargs,
) -> tuple[GaussianLikelihoodConfig, MultiChannelNMConfig, NMLikelihoodConfig]:
    """Get the likelihood configuration for split models.
    
    Parameters
    ----------
    ...
    
    Returns
    -------
    tuple[GaussianLikelihoodConfig, MultiChannelNMConfig, NMLikelihoodConfig]
        The likelihoods and noise models configurations.
    """
    # gaussian likelihood
    if kwargs["loss_type"] in ["musplit", "denoisplit_musplit"]:
        gaussian_lik_config = GaussianLikelihoodConfig(
            predict_logvar=kwargs["predict_logvar"],
            logvar_lowerbound=-5.0,
        )
    else:
        gaussian_lik_config = None
    # noise model likelihood
    if kwargs["loss_type"] in ["denoisplit", "denoisplit_musplit"]:
        gmm_list = []
        for NM_path in kwargs["nm_paths"]:
            gmm_list.append(
                GaussianMixtureNMConfig(
                    model_type="GaussianMixtureNoiseModel",
                    path=NM_path,
                )
            )
        noise_model_config = MultiChannelNMConfig(noise_models=gmm_list)
        nm_lik_config = NMLikelihoodConfig(
            data_mean=kwargs["data_stats"][0],
            data_std=kwargs["data_stats"][1],
        )
    else:
        noise_model_config = None
        nm_lik_config = None

    return gaussian_lik_config, noise_model_config, nm_lik_config


def get_loss_config(**kwargs) -> LVAELossConfig:
    """Get the loss configuration for the split model.
    
    Parameters
    ----------
    ...
    
    Returns
    -------
    LVAELossConfig
        The loss configuration.
    """
    return LVAELossConfig(
        loss_type=kwargs["loss_type"],
        kl_params=KLLossConfig(),
    )
    

def get_training_config(**kwargs) -> TrainingConfig:
    """Get the training configuration.
    
    Parameters
    ----------
    ...
    
    Returns
    -------
    TrainingConfig
        The training configuration.
    """
    return TrainingConfig(
        num_epochs=kwargs["num_epochs"],
        precision="16-mixed",
        logger="wandb",
        gradient_clip_algorithm="value",
        grad_clip_norm_value=0.5,
        lr=kwargs["lr"],
        lr_scheduler_patience=kwargs["lr_scheduler_patience"],
    )


def get_lr_scheduler_config(**kwargs) -> LrSchedulerModel:
    return LrSchedulerModel(
        name="ReduceLROnPlateau",
        parameters={
            "mode": "min",
            "factor": 0.5,
            "patience": kwargs["lr_scheduler_patience"],
            "verbose": True,
            "min_lr": 1e-12,
        },
    )
    

def get_optimizer_config(**kwargs) -> OptimizerModel:
    return OptimizerModel(
        name="Adamax",
        parameters={
            "lr": kwargs["lr"],
            "weight_decay": 0,
        },
    )


def get_algorithm_config(
    algorithm: Literal["muspit", "denoisplit"],
    loss_config: LVAELossConfig,
    model_config: LVAEModel,
    gaussian_lik_config: GaussianLikelihoodConfig,
    nm_config: MultiChannelNMConfig,
    nm_lik_config: NMLikelihoodConfig,
    opt_config: OptimizerModel,
    lr_scheduler_config: LrSchedulerModel,    
) -> VAEAlgorithmConfig:
    """Instantiate the split algorithm config.
    
    Parameters
    ----------
    algorithm : Literal["muspit", "denoisplit"]
        The algorithm type.
    loss_config : LVAELossConfig
        The loss configuration.
    model_config : LVAEModel
        The LVAE model configuration.
    gaussian_lik_config : GaussianLikelihoodConfig
        The Gaussian likelihood configuration.
    nm_config : MultiChannelNMConfig
        The noise model configuration.
    nm_lik_config : NMLikelihoodConfig
        The noise model likelihood configuration.
    opt_config : OptimizerModel
        The optimizer configuration.
    lr_scheduler_config : LrSchedulerModel
        The learning rate scheduler configuration.

    Returns
    -------
    VAEAlgorithmConfig
        The split algorithm configuration.
    """
    return VAEAlgorithmConfig(
        algorithm_type="vae",
        algorithm=algorithm,
        loss=loss_config,
        model=model_config,
        gaussian_likelihood=gaussian_lik_config,
        noise_model=nm_config,
        noise_model_likelihood=nm_lik_config,
        optimizer=opt_config,
        lr_scheduler=lr_scheduler_config,
    )