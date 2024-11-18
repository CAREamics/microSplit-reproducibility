from typing import List, Optional

from careamics.lvae_training.dataset import DatasetConfig, DataSplitType, DataType

# from datasets.ht_iba1_ki64_2023 import SubDsetType


class PunctaRemovalConfig(DatasetConfig):
    channel_list: List[str] = ['puncta','foreground']
    background_values: List[int] = [0, 0]
    data_path: Optional[str] = None # TODO: add to `DatasetConfig` in `lvae_training.dataset`
    
DEFAULT_CONFIG = PunctaRemovalConfig(
    datasplit_type=DataSplitType.Train,
    image_size=(64, 64),
    grid_size=32,
    data_type=DataType.HTIba1Ki67,
    poisson_noise_factor=-1,
    enable_gaussian_noise=False,
    synthetic_gaussian_scale=6675,
    input_has_dependant_noise=True,
    multiscale_lowres_count=1,
    use_one_mu_std=True,
    train_aug_rotate=True,
    target_separate_normalization=True,
    input_is_sum=True,
    padding_kwargs={"mode": "reflect"},
    overlapping_padding_kwargs={"mode": "reflect"},
)

def get_train_data_configs() -> tuple[PunctaRemovalConfig, PunctaRemovalConfig]:
    """Get the data configurations to use at training time."""
    train_data_config = DEFAULT_CONFIG.model_copy()
    val_data_config = DEFAULT_CONFIG.model_copy(
        update=dict(
            datasplit_type=DataSplitType.Val,
            allow_generation=False,  # No generation during validation
            enable_random_cropping=False,  # No random cropping on validation.
        )
    )
    return train_data_config, val_data_config
    

def get_eval_data_configs(
) -> tuple[PunctaRemovalConfig, PunctaRemovalConfig, PunctaRemovalConfig]:
    """Get the data configurations to use at evaluation time."""
    train_data_config = DEFAULT_CONFIG.model_copy()
    val_data_config = DEFAULT_CONFIG.model_copy(
        update=dict(
            datasplit_type=DataSplitType.Val,
            allow_generation=False,  # No generation during validation
            enable_random_cropping=False,  # No random cropping on validation.
        )
    )
    test_data_config = val_data_config.model_copy(
        update=dict(
            datasplit_type=DataSplitType.Test,
        )
    )
    return (
        train_data_config, 
        val_data_config,
        test_data_config,
    )