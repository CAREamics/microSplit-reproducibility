import os

import numpy as np

import nd2
from nis2pyr.reader import read_nd2file

from careamics.lvae_training.dataset import DataSplitType
from careamics.lvae_training.dataset.utils.data_utils import get_datasplit_tuples


def load_7D(fpath):
    print(f"Loading from {fpath}")
    with nd2.ND2File(fpath) as nd2file:
        data = read_nd2file(nd2file)
    return data


def load_filtered_zstacks(fpath, zstart=20, zstop=44):
    data = load_7D(fpath)
    return data[0, :, zstart:zstop, ..., 0]


def datafiles():
    return ["20240725/WTC11_WT_DIV25_3_1_0001.nd2"]


def get_train_val_data(
    data_config,
    datadir,
    datasplit_type: DataSplitType,
    val_fraction=None,
    test_fraction=None,
    **kwargs,
):
    fnames = datafiles()
    zstart = data_config.z_start
    zstop = data_config.z_stop
    ch_list = data_config.channel_idx_list
    fpaths = [os.path.join(datadir, x) for x in fnames]
    data_arr = [
        load_filtered_zstacks(fpath, zstart=zstart, zstop=zstop)[:, :, ch_list]
        for fpath in fpaths
    ]
    if len(data_arr) == 1:
        data = data_arr[0]
    else:
        raise Exception("Multiple files not supported")

    idx_list = np.random.RandomState(955).permutation(len(data))
    train_idx, val_idx, test_idx = get_datasplit_tuples(
        val_fraction, test_fraction, len(data)
    )
    train_idx = idx_list[train_idx]
    val_idx = idx_list[val_idx]
    test_idx = idx_list[test_idx]

    if datasplit_type == DataSplitType.All:
        data = data.astype(np.float32)
    elif datasplit_type == DataSplitType.Train:
        print("train_idx", train_idx)
        data = data[train_idx].astype(np.float32)
    elif datasplit_type == DataSplitType.Val:
        print("val_idx", val_idx)
        data = data[val_idx].astype(np.float32)
    elif datasplit_type == DataSplitType.Test:
        print("test_idx", test_idx)
        data = data[test_idx].astype(np.float32)
    else:
        raise Exception("invalid datasplit")

    data = np.transpose(data, (0, 1, 3, 4, 2))
    return data
