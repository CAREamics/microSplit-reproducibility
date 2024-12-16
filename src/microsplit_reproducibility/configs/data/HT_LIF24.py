from typing import Literal

from careamics.lvae_training.dataset import DatasetConfig, DataSplitType, DataType

CH_IDX_LIST = [1, 2, 3, 17]


class NikolaDataConfig(DatasetConfig):
    dset_type: Literal[
        "high", "mid", "low", "verylow", "2ms", "3ms", "5ms", "20ms", "500ms"
    ]
    # TODO: add description
    
    channel_idx_list: list[Literal[1, 2, 3, 17]]
    # TODO: add description


def get_data_configs(
    dset_type: Literal["high", "mid", "low", "verylow", "2ms", "3ms", "5ms", "20ms", "500ms"],
    channel_idx_list: list = [1, 2, 3, 17],
) -> tuple[NikolaDataConfig, NikolaDataConfig, NikolaDataConfig]:
    """Get the data configurations to use at training time.
    
    Parameters
    ----------
    dset_type : Literal["high", "mid", "low", "verylow", "2ms", "3ms", "5ms", "20ms", "500ms"]
        The dataset type to use.
    channel_idx_list : list[Literal[1, 2, 3, 17]]
        The channel indices to use.
    
    Returns
    -------
    tuple[NikolaDataConfig, NikolaDataConfig]
        The train, validation and test data configurations.
    """
    train_data_config = NikolaDataConfig(
        data_type=DataType.NicolaData,
        dset_type=dset_type,
        datasplit_type=DataSplitType.Train,
        image_size=[64, 64],
        grid_size=32,
        channel_idx_list=channel_idx_list,
        num_channels=len(channel_idx_list),
        input_idx=len(channel_idx_list) - 1,
        target_idx_list=list(range(len(channel_idx_list) - 1)),
        multiscale_lowres_count=3,
        poisson_noise_factor=-1,
        enable_gaussian_noise=False,
        synthetic_gaussian_scale=100,
        input_has_dependant_noise=True,
        use_one_mu_std=True,
        train_aug_rotate=True,
        target_separate_normalization=True,
        input_is_sum=False,
        padding_kwargs={"mode": "reflect"},
        overlapping_padding_kwargs={"mode": "reflect"},
    )
    val_data_config = train_data_config.model_copy(
        update=dict(
            datasplit_type=DataSplitType.Val,
            allow_generation=False,  # No generation during validation
            enable_random_cropping=False,  # No random cropping on validation.
        )
    )
    test_data_config = val_data_config.model_copy(
        update=dict(datasplit_type=DataSplitType.Test,)
    )
    return train_data_config, val_data_config, test_data_config
