from careamics.lvae_training.dataset import DatasetConfig, DataSplitType, DataType

from datasets.multifile_raw_dataloader import SubDsetType


class DAO3CHConfig(DatasetConfig):
    subdset_type: SubDsetType
    channel_idx_list: list[int]


def get_data_configs() -> tuple[DAO3CHConfig, DAO3CHConfig]:
    channel_idx_list = [0, 1, 2, 3]
    num_channels = len(channel_idx_list)

    train_data_config = DAO3CHConfig(
        datasplit_type=DataSplitType.Train,
        image_size=[64, 64],
        grid_size=32,
        num_channels=2, # TODO hardcoded because only 2 channels are used at the same time
        data_type=DataType.Dao3Channel,
        subdset_type=SubDsetType.MultiChannel,
        channel_idx_list=channel_idx_list,
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

    val_data_config = train_data_config.model_copy(
        update=dict(
            datasplit_type=DataSplitType.Val,
            allow_generation=False,  # No generation during validation
            enable_random_cropping=False,  # No random cropping on validation.
        )
    )
    
    test_data_config = train_data_config.model_copy(
        update=dict(
            datasplit_type=DataSplitType.Val,
            allow_generation=False,  # No generation during validation
            enable_random_cropping=False,  # No random cropping on validation.
        )
    )

    return train_data_config, val_data_config, test_data_config
