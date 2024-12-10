from typing import List, Optional

from careamics.lvae_training.dataset import DatasetConfig, DataSplitType, DataType

DEFAULT_CONFIG = DatasetConfig(
    datasplit_type=DataSplitType.Train,
    image_size=(64, 64),
    grid_size=32,
    data_type=DataType.Care3D,
    poisson_noise_factor=-1,
    enable_gaussian_noise=False,
    multiscale_lowres_count=1,
    use_one_mu_std=True,
    train_aug_rotate=True,
    target_separate_normalization=True,
    input_is_sum=False,
    num_channels=3,
    depth3D=9,
    mode_3D=True,
    subdset_type='zebrafish',
    padding_kwargs={"mode": "reflect"},
    overlapping_padding_kwargs={"mode": "reflect"},
)

def get_train_data_configs(depth3D=9,subdset_type='zebrafish') -> tuple[DatasetConfig, DatasetConfig]:
    """Get the data configurations to use at training time."""
    train_data_config = DEFAULT_CONFIG.model_copy()
    val_data_config = DEFAULT_CONFIG.model_copy(
        update=dict(
            datasplit_type=DataSplitType.Val,
            allow_generation=False,  # No generation during validation
            enable_random_cropping=False,  # No random cropping on validation.
        )
    )

    train_data_config.depth3D = depth3D
    val_data_config.depth3D = depth3D
    train_data_config.subdset_type = subdset_type
    val_data_config.subdset_type = subdset_type

    return train_data_config, val_data_config
    

def get_eval_data_configs(
) -> tuple[DatasetConfig, DatasetConfig, DatasetConfig]:
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