from enum import Enum

from careamics.lvae_training.dataset import DatasetConfig, DataSplitType, DataType

from datasets.multifile_raw_dataloader import SubDsetType


class Sox2GolgiV2ChannelList(Enum):
    GT_Cy5 = "GT_Cy5"
    GT_TRITC = "GT_TRITC"
    GT_555_647 = "555-647"


class Sox2GolgiV2Config(DatasetConfig):
    subdset_type: SubDsetType
    channel_idx_list: list[Sox2GolgiV2ChannelList]


def get_data_configs() -> tuple[Sox2GolgiV2Config, Sox2GolgiV2Config]:
    channel_idx_list = [
        Sox2GolgiV2ChannelList.GT_Cy5,
        Sox2GolgiV2ChannelList.GT_TRITC,
        Sox2GolgiV2ChannelList.GT_555_647,
    ]
    train_data_config = Sox2GolgiV2Config(
        datasplit_type=DataSplitType.Train,
        data_type=DataType.TavernaSox2GolgiV2,
        num_channels=len(channel_idx_list),
        image_size=64,
        subdset_type=SubDsetType.MultiChannel,
        channel_idx_list=channel_idx_list,
        multiscale_lowres_count=3,
        input_idx=2,
        target_idx_list=[0, 1],
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
