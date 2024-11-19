from careamics.lvae_training.dataset import DatasetConfig, DataSplitType, DataType


def get_data_configs() -> tuple[DatasetConfig, DatasetConfig]:
    train_data_config = DatasetConfig(
        data_type=DataType.BioSR_MRC,
        datasplit_type=DataSplitType.Train,
        image_size=64,
        num_channels=2,
        ch1_fname="ER/GT_all.mrc",
        ch2_fname="CCPs/GT_all.mrc",
        multiscale_lowres_count=1,
        enable_gaussian_noise=True,
        synthetic_gaussian_scale=5100,
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
