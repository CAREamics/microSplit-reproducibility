import os
from enum import Enum

import numpy as np
from careamics.lvae_training.dataset import DataSplitType
from careamics.lvae_training.dataset.utils.data_utils import (
    get_datasplit_tuples,
    load_tiff,
)

# from czifile import imread as imread_czi


class SNR(Enum):
    Low = "low"
    High = "high"


class SubDsetType(Enum):
    OnlyIba1 = "Iba1"
    Iba1Ki64 = "Iba1_Ki67"
    OnlyIba1P30 = "Iba1NucPercent30"
    OnlyIba1P50 = "Iba1NucPercent50"
    OnlyIba1P70 = "Iba1NucPercent70"


# def get_iba1_ki67_files():
#     return [f'{i}.czi' for i in range(1, 31)]

# def get_iba1_only_files():
#     return [f'Iba1only_{i}.czi' for i in range(1, 16)]


def get_iba1_ki67_files(snrtype: SNR):
    return [f"Iba1_Ki67_{snrtype.value}_{i}.czi" for i in range(1, 16)]


def get_iba1_only_files(snrtype: SNR):
    if snrtype == SNR.Low:
        return [f"Iba1_{snrtype.value}_{i}.czi" for i in range(1, 16)]
    elif snrtype == SNR.High:
        return [f"Iba1_{i}.czi" for i in range(1, 16)]
    else:
        raise Exception(f"Invalid snrtype: {snrtype}")


def load_czi(fpaths):
    imgs = []
    for fpath in fpaths:
        img = imread_czi(fpath)
        assert img.shape[3] == 1
        img = np.swapaxes(img, 0, 3)
        # the first dimension of img stored in imgs will have dim of 1, where the contenation will happen
        imgs.append(img)
    return np.concatenate(imgs, axis=0)


def get_subdir(subdset_type, snrtype):
    if subdset_type in [
        SubDsetType.OnlyIba1P30,
        SubDsetType.OnlyIba1P50,
        SubDsetType.OnlyIba1P70,
    ]:
        subdset_type = SubDsetType.OnlyIba1
    return f"{subdset_type.value}_{snrtype.value}"


def get_train_val_data(
    data_config,
    datadir,
    datasplit_type: DataSplitType,
    val_fraction=None,
    test_fraction=None,
    **kwargs,
):
    dset_subtype = data_config.subdset_type
    subdir = get_subdir(dset_subtype, data_config.snrtype)

    if dset_subtype in [
        SubDsetType.OnlyIba1,
        SubDsetType.OnlyIba1P30,
        SubDsetType.OnlyIba1P50,
        SubDsetType.OnlyIba1P70,
    ]:
        fnames = get_iba1_only_files(data_config.snrtype)
    elif dset_subtype == SubDsetType.Iba1Ki64:
        fnames = get_iba1_ki67_files(data_config.snrtype)
    else:
        raise Exception(f"Invalid dset subtype: {dset_subtype}")

    train_idx, val_idx, test_idx = get_datasplit_tuples(
        val_fraction, test_fraction, len(fnames)
    )
    if datasplit_type == DataSplitType.All:
        fpaths = [os.path.join(datadir, subdir, x) for x in fnames]
        data = load_czi(fpaths)
    elif datasplit_type == DataSplitType.Train:
        print(train_idx)
        fnames = [fnames[i] for i in train_idx]
        fpaths = [os.path.join(datadir, subdir, x) for x in fnames]
        data = load_czi(fpaths)
    elif datasplit_type == DataSplitType.Val:
        print(val_idx)
        fnames = [fnames[i] for i in val_idx]
        fpaths = [os.path.join(datadir, subdir, x) for x in fnames]
        data = load_czi(fpaths)
    elif datasplit_type == DataSplitType.Test:
        print(test_idx)
        fnames_iba1 = [fnames[i] for i in test_idx]
        fpaths_iba1 = [os.path.join(datadir, subdir, x) for x in fnames_iba1]

        # it contains iba1/iba1ki67 and DAPI.
        data = load_czi(fpaths_iba1)

        data_nuc = None
        if dset_subtype in [
            SubDsetType.OnlyIba1P30,
            SubDsetType.OnlyIba1P50,
            SubDsetType.OnlyIba1P70,
        ]:
            datadir_nuc = os.path.join(
                datadir, SubDsetType.OnlyIba1.value, f"synthetic_test/{dset_subtype}"
            )
            print("Loading nucleus from", datadir_nuc)
            fnames_nuc = sorted(os.listdir(datadir_nuc))
            fpaths_nuc = [os.path.join(datadir_nuc, x) for x in fnames_nuc]
            fpaths_nuc = [fpath for fpath in fpaths_nuc if not os.path.isdir(fpath)]
            data_nuc = np.concatenate(
                [load_tiff(fpath_)[None] for fpath_ in fpaths_nuc], axis=0
            )[..., None]
            data = np.tile(data[..., 1:], (len(data_nuc), 1, 1, 1))
            data = np.concatenate([data_nuc, data], axis=3)

    else:
        raise Exception("invalid datasplit")

    # fpaths = [os.path.join(datadir, dset_subtype, x) for x in fnames]
    # data = load_czi(fpaths)
    print("Loaded from", SubDsetType(dset_subtype), datadir, data.shape)
    if dset_subtype == SubDsetType.Iba1Ki64:
        # We just need the combined channel. we don't need the nuclear channel.
        # in order for the whole setup to work well, I'm just copying the channel twice.
        # when creating the input, the average of these channels will still be exactly this channel, which is what we want.
        # we want this channel as input to the network.
        # Note that mean and the stdev used to normalize this data will be different, but we can try to do that initially.
        data = data[..., 1:]
        data = np.tile(data, (1, 1, 1, 2))
    return data
