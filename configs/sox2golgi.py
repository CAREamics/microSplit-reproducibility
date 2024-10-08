from careamics.lvae_training.dataset import DatasetConfig, DataSplitType, DataType

from datasets.multifile_raw_dataloader import SubDsetType
from datasets.sox2golgi import Sox2GolgiChannelList


class Sox2GolgiConfig(DatasetConfig):
    subdset_type: SubDsetType
    channel_1: Sox2GolgiChannelList
    channel_2: Sox2GolgiChannelList


def get_data_configs() -> tuple[Sox2GolgiConfig, Sox2GolgiConfig]:
    train_data_config = Sox2GolgiConfig(
        datasplit_type=DataSplitType.Train,
        data_type=DataType.TavernaSox2Golgi,
        image_size=64,
        subdset_type=SubDsetType.TwoChannel,
        channel_1=Sox2GolgiChannelList.Sox2,
        channel_2=Sox2GolgiChannelList.Golgi,
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

    return train_data_config, val_data_config
