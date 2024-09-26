from careamics.lvae_training.dataset import DatasetConfig, DataSplitType, DataType


CH_IDX_LIST = [0, 1, 2, 3]
DSET_TYPE = "500ms"


def get_data_configs() -> tuple[DatasetConfig, DatasetConfig]:
    train_data_config = DatasetConfig(
        datasplit_type=DataSplitType.Train,
        image_size=64,
        num_channels=len(CH_IDX_LIST),
        input_idx=len(CH_IDX_LIST) - 1,
        target_idx_list=list(range(len(CH_IDX_LIST) - 1)),
        multiscale_lowres_count=1,
        data_type=DataType.NicolaData,
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

    return train_data_config, val_data_config
