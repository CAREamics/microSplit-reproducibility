import os
from enum import Enum

import numpy as np
from skimage.io import imread
from careamics.lvae_training.dataset.utils.data_utils import get_datasplit_tuples
from careamics.lvae_training.dataset.types import DataSplitType
from careamics.lvae_training.dataset.multifile_dataset import (
    TwoChannelData,
    MultiChannelData,
)


class SubDsetType(Enum):
    TwoChannel = 0
    OneChannel = 1
    MultiChannel = 2


def load_tiff(path):
    """
    Returns a 4d numpy array: num_imgs*h*w*num_channels
    """
    data = imread(path, plugin="tifffile")
    return data


def subset_data(dataA, dataB, dataidx_list):
    dataidx_list = sorted(dataidx_list)
    subset_dataA = []
    subset_dataB = [] if dataB is not None else None
    cur_dataidx = 0
    cumulative_datacount = 0
    for arr_idx in range(len(dataA)):
        for data_idx in range(len(dataA[arr_idx])):
            cumulative_datacount += 1
            if dataidx_list[cur_dataidx] == cumulative_datacount - 1:
                subset_dataA.append(dataA[arr_idx][data_idx : data_idx + 1])
                if dataB is not None:
                    subset_dataB.append(dataB[arr_idx][data_idx : data_idx + 1])
                cur_dataidx += 1
            if cur_dataidx >= len(dataidx_list):
                break
        if cur_dataidx >= len(dataidx_list):
            break
    return subset_dataA, subset_dataB


def get_train_val_data(
    datadir,
    data_config,
    datasplit_type: DataSplitType,
    get_multi_channel_files_fn,
    load_data_fn=None,
    val_fraction=None,
    test_fraction=None,
    explicit_val_idx=None,
    explicit_test_idx=None,
):
    print("")
    dset_subtype = data_config.subdset_type
    if load_data_fn is None:
        load_data_fn = load_tiff

    if dset_subtype == SubDsetType.TwoChannel:
        fnamesA, fnamesB = get_multi_channel_files_fn()
        fpathsA = [os.path.join(datadir, x) for x in fnamesA]
        fpathsB = [os.path.join(datadir, x) for x in fnamesB]
        dataA = [load_data_fn(fpath) for fpath in fpathsA]
        dataB = [load_data_fn(fpath) for fpath in fpathsB]
    elif dset_subtype == SubDsetType.OneChannel:
        fnamesmixed = get_multi_channel_files_fn()
        fpathsmixed = [os.path.join(datadir, x) for x in fnamesmixed]
        fpathsA = fpathsB = fpathsmixed
        dataA = [load_data_fn(fpath) for fpath in fpathsmixed]
        # Note that this is important. We need to ensure that the sum of the two channels is the same as sum of these two channels.
        dataA = [x / 2 for x in dataA]
        dataB = [x.copy() for x in dataA]
    elif dset_subtype == SubDsetType.MultiChannel:
        fnamesA = get_multi_channel_files_fn()
        fpathsA = [os.path.join(datadir, x) for x in fnamesA]
        dataA = [load_data_fn(fpath) for fpath in fpathsA]
        fnamesB = None
        fpathsB = None
        dataB = None

    if dataB is not None:
        assert len(dataA) == len(dataB)
        for i in range(len(dataA)):
            assert (
                dataA[i].shape == dataB[i].shape
            ), f"{dataA[i].shape} != {dataB[i].shape}, {fpathsA[i]} != {fpathsB[i]} in shape"

            if len(dataA[i].shape) == 2:
                dataA[i] = dataA[i][None]
                dataB[i] = dataB[i][None]

    count = np.sum([x.shape[0] for x in dataA])
    framewise_fpathsA = []
    for onedata_A, onepath_A in zip(dataA, fpathsA):
        framewise_fpathsA += [onepath_A] * onedata_A.shape[0]

    framewise_fpathsB = None
    if dataB is not None:
        framewise_fpathsB = []
        for onedata_B, onepath_B in zip(dataB, fpathsB):
            framewise_fpathsB += [onepath_B] * onedata_B.shape[0]

    # explicit datasplit
    if explicit_val_idx is not None:
        assert explicit_test_idx is not None
        train_idx = [
            i
            for i in range(count)
            if i not in explicit_val_idx and i not in explicit_test_idx
        ]
        val_idx = explicit_val_idx
        test_idx = explicit_test_idx
        if datasplit_type == DataSplitType.Val:
            print("Explicit datasplit Val", val_idx)
        elif datasplit_type == DataSplitType.Test:
            print("Explicit datasplit Test", test_idx)
        elif datasplit_type == DataSplitType.Train:
            print("Explicit datasplit Train", train_idx)
    else:
        train_idx, val_idx, test_idx = get_datasplit_tuples(
            val_fraction, test_fraction, count
        )

    if datasplit_type == DataSplitType.All:
        pass
    elif datasplit_type == DataSplitType.Train:
        # print(train_idx)
        dataA, dataB = subset_data(dataA, dataB, train_idx)
        framewise_fpathsA = [framewise_fpathsA[i] for i in train_idx]
        if dataB is not None:
            framewise_fpathsB = [framewise_fpathsB[i] for i in train_idx]
    elif datasplit_type == DataSplitType.Val:
        # print(val_idx)
        dataA, dataB = subset_data(dataA, dataB, val_idx)
        framewise_fpathsA = [framewise_fpathsA[i] for i in val_idx]
        if dataB is not None:
            framewise_fpathsB = [framewise_fpathsB[i] for i in val_idx]
    elif datasplit_type == DataSplitType.Test:
        # print(test_idx)
        dataA, dataB = subset_data(dataA, dataB, test_idx)
        framewise_fpathsA = [framewise_fpathsA[i] for i in test_idx]
        if dataB is not None:
            framewise_fpathsB = [framewise_fpathsB[i] for i in test_idx]
    else:
        raise Exception("invalid datasplit")

    if hasattr(data_config, "channel_idx_list"):
        assert isinstance(data_config.channel_idx_list, list) or isinstance(
            data_config.channel_idx_list, tuple
        ), "channel_idx_list should be a list"
        assert all(
            [
                isinstance(data_config.channel_idx_list[i], int)
                or isinstance(data_config.channel_idx_list[i], str)
                for i in range(len(data_config.channel_idx_list))
            ]
        ), f"Invalid channel_idx_list {data_config.channel_idx_list}"

        if isinstance(data_config.channel_idx_list[0], int):
            print("Selecting channels", data_config.channel_idx_list)
            dataA = [x[..., data_config.channel_idx_list] for x in dataA]
            if dataB is not None:
                dataB = [x[..., data_config.channel_idx_list] for x in dataB]
        else:
            print(
                "Warning: channel_idx_list is not a list of integers, but a list of strings. No selection of channels is done"
            )

    if dset_subtype == SubDsetType.MultiChannel:
        data = MultiChannelData(dataA, paths=framewise_fpathsA)
    else:
        data = TwoChannelData(
            dataA, dataB, paths_data1=framewise_fpathsA, paths_data2=framewise_fpathsB
        )
    print(
        "Loaded from",
        SubDsetType(dset_subtype),
        datadir,
        f"{len(data)}/{count} frames",
    )
    return data
