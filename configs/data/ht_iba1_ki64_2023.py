from typing import Optional

from careamics.lvae_training.dataset import DatasetConfig, DataSplitType, DataType

from datasets.ht_iba1_ki64_2023 import SubDsetType


class HTIBA1Ki64Config(DatasetConfig):
    subdset_type: SubDsetType
    data_path: Optional[str] = None # TODO: add to `DatasetConfig` in `lvae_training.dataset`
    
DEFAULT_CONFIG = HTIBA1Ki64Config(
    datasplit_type=DataSplitType.Train,
    image_size=(64, 64),
    grid_size=32,
    data_type=DataType.HTIba1Ki67,
    subdset_type=SubDsetType.OnlyIba1,
    poisson_noise_factor=-1,
    enable_gaussian_noise=False,
    synthetic_gaussian_scale=6675,
    input_has_dependant_noise=True,
    multiscale_lowres_count=3,
    use_one_mu_std=True,
    train_aug_rotate=True,
    target_separate_normalization=True,
    input_is_sum=False,
    padding_kwargs={"mode": "reflect"},
    overlapping_padding_kwargs={"mode": "reflect"},
)

def get_train_data_configs() -> tuple[HTIBA1Ki64Config, HTIBA1Ki64Config]:
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
    test_subdset_type: SubDsetType
) -> tuple[HTIBA1Ki64Config, HTIBA1Ki64Config, HTIBA1Ki64Config]:
    """Get the data configurations to use at evaluation time."""
    test_subdset_type = SubDsetType(test_subdset_type)
    # assert test_subdset_type in [
    #     SubDsetType.OnlyIba1P30,
    #     SubDsetType.OnlyIba1P50,
    #     SubDsetType.OnlyIba1P70,
    #     SubDsetType.OnlyIba1,
    # ], f"Invalid test_subdset_type: {test_subdset_type}"
    
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
            subdset_type=test_subdset_type,
        )
    )
    return (
        train_data_config, 
        val_data_config,
        test_data_config,
    )