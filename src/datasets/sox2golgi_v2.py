import os
from functools import partial

import numpy as np
from careamics.lvae_training.dataset import DataSplitType
from nd2reader import ND2Reader

from .multifile_raw_dataloader import (
    get_train_val_data as get_train_val_data_multichannel,
)


def get_start_end_index(key):
    """
    Few start and end frames are not good in some of the files. So, we need to exclude them.
    """
    start_index_dict = {
        "Test1_Slice1/1.nd2": 8,
        "Test1_Slice1/2.nd2": 1,
        "Test1_Slice1/3.nd2": 3,
        "Test1_Slice2_a/4.nd2": 10,
        "Test1_Slice2_a/5.nd2": 10,
        "Test1_Slice2_a/6.nd2": 10,
        "Test1_Slice2_b/7.nd2": 1,
        "Test1_Slice3_b/4.nd2": 1,
        "Test1_Slice3_b/5.nd2": 1,
        "Test1_Slice3_b/6.nd2": 1,
        "Test1_Slice4_a/1.nd2": 1,
        "Test1_Slice4_a/2.nd2": 1,
        "Test1_Slice4_a/3.nd2": 1,
        "Test1_Slice4_b/4.nd2": 1,
        "Test1_Slice4_b/5.nd2": 1,
        "Test1_Slice4_b/6.nd2": 1,
    }
    # excluding this index
    end_index_dict = {
        "Test1_Slice2_b/7.nd2": 18,
        "Test1_Slice2_b/8.nd2": 18,
        "Test1_Slice2_b/9.nd2": 18,
        "Test1_Slice3_a/1.nd2": 15,
        "Test1_Slice3_a/2.nd2": 15,
        "Test1_Slice3_a/3.nd2": 15,
        "Test1_Slice3_b/4.nd2": 18,
        "Test1_Slice3_b/5.nd2": 18,
        "Test1_Slice3_b/6.nd2": 18,
        "Test1_Slice4_a/1.nd2": 19,
        "Test1_Slice4_a/2.nd2": 19,
        "Test1_Slice4_a/3.nd2": 19,
    }
    return start_index_dict.get(key), end_index_dict.get(key)


def load_nd2(fpath, channel_names=None, multiplicative_factor=1):
    fname = os.path.basename(fpath)
    parent_dir = os.path.basename(os.path.dirname(fpath))
    key = os.path.join(parent_dir, fname)
    start_z, end_z = get_start_end_index(key)
    with ND2Reader(fpath) as reader:
        data = []
        if start_z is None:
            start_z = 0
        if end_z is None:
            end_z = reader.metadata["total_images_per_channel"]

        all_channels = reader.metadata["channels"]
        relevant_channel_indices = [all_channels.index(c) for c in channel_names]

        for z in range(start_z, end_z):
            channels = []
            for c in relevant_channel_indices:
                img = reader.get_frame_2D(c=c, z=z)
                img = img * multiplicative_factor
                channels.append(img[..., None])
            img = np.concatenate(channels, axis=-1)
            data.append(img[None])
        data = np.concatenate(data, axis=0)
    return data


def get_files():
    rel_fpaths = []
    rel_fpaths += ["Test1_Slice1/1.nd2", "Test1_Slice1/2.nd2", "Test1_Slice1/3.nd2"]
    rel_fpaths += [
        "Test1_Slice2_a/4.nd2",
        "Test1_Slice2_a/5.nd2",
        "Test1_Slice2_a/6.nd2",
    ]
    rel_fpaths += [
        "Test1_Slice2_b/7.nd2",
        "Test1_Slice2_b/8.nd2",
        "Test1_Slice2_b/9.nd2",
    ]
    rel_fpaths += [
        "Test1_Slice3_a/1.nd2",
        "Test1_Slice3_a/2.nd2",
        "Test1_Slice3_a/3.nd2",
    ]
    rel_fpaths += [
        "Test1_Slice3_b/4.nd2",
        "Test1_Slice3_b/5.nd2",
        "Test1_Slice3_b/6.nd2",
    ]
    rel_fpaths += [
        "Test1_Slice4_a/1.nd2",
        "Test1_Slice4_a/2.nd2",
        "Test1_Slice4_a/3.nd2",
    ]
    rel_fpaths += [
        "Test1_Slice4_b/4.nd2",
        "Test1_Slice4_b/5.nd2",
        "Test1_Slice4_b/6.nd2",
    ]
    return rel_fpaths


def get_train_val_data(
    data_config,
    datadir,
    datasplit_type: DataSplitType,
    val_fraction=None,
    test_fraction=None,
    **kwargs,
):
    if "channel_idx_list" in data_config.__fields__.keys():
        channel_names = data_config.channel_idx_list
    else:
        assert "channel_1" in data_config.__fields__.keys()
        assert "channel_2" in data_config.__fields__.keys()
        channel_names = [data_config.channel_1, data_config.channel_2]
    print(
        f"Loading data from {datadir} with channel names {channel_names}, datasplit_type {DataSplitType(datasplit_type)}"
    )
    return get_train_val_data_multichannel(
        datadir,
        data_config,
        datasplit_type,
        get_files,
        load_data_fn=partial(load_nd2, channel_names=channel_names),
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )
