# from microssim import MicroMS3IM, MicroSSIM
# ssim
from collections import defaultdict

import numpy as np
import torch

from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from skimage.metrics import structural_similarity


from microssim import MicroMS3IM, MicroSSIM


def allow_numpy(func):
    """
    All optional arguements are passed as is. positional arguments are checked. if they are numpy array,
    they are converted to torch Tensor.
    """

    def numpy_wrapper(*args, **kwargs):
        new_args = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                arg = torch.Tensor(arg)
            new_args.append(arg)
        new_args = tuple(new_args)

        output = func(*new_args, **kwargs)
        return output

    return numpy_wrapper


@allow_numpy
def range_invariant_multiscale_ssim(gt_, pred_):
    """
    Computes range invariant multiscale ssim for one channel.
    This has the benefit that it is invariant to scalar multiplications in the prediction.
    """

    shape = gt_.shape
    gt_ = torch.Tensor(gt_.reshape((shape[0], -1)))
    pred_ = torch.Tensor(pred_.reshape((shape[0], -1)))
    gt_ = zero_mean(gt_)
    pred_ = zero_mean(pred_)
    pred_ = fix(gt_, pred_)
    pred_ = pred_.reshape(shape)
    gt_ = gt_.reshape(shape)

    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(
        data_range=gt_.max() - gt_.min()
    )
    return ms_ssim(torch.Tensor(pred_[:, None]), torch.Tensor(gt_[:, None])).item()


def compute_multiscale_ssim(gt_, pred_, range_invariant=True):
    """
    Computes multiscale ssim for each channel.
    Args:
    gt_: ground truth image with shape (N, H, W, C)
    pred_: predicted image with shape (N, H, W, C)
    range_invariant: whether to use range invariant multiscale ssim
    """
    ms_ssim_values = {i: None for i in range(gt_.shape[-1])}
    for ch_idx in range(gt_.shape[-1]):
        tar_tmp = gt_[..., ch_idx]
        pred_tmp = pred_[..., ch_idx]
        if range_invariant:
            ms_ssim_values[ch_idx] = [
                range_invariant_multiscale_ssim(tar_tmp[i : i + 1], pred_tmp[i : i + 1])
                for i in range(tar_tmp.shape[0])
            ]
        else:
            ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(
                data_range=tar_tmp.max() - tar_tmp.min()
            )
            ms_ssim_values[ch_idx] = [
                ms_ssim(
                    torch.Tensor(pred_tmp[i : i + 1, None]),
                    torch.Tensor(tar_tmp[i : i + 1, None]),
                ).item()
                for i in range(tar_tmp.shape[0])
            ]

    output = [
        (np.mean(ms_ssim_values[i]), np.std(ms_ssim_values[i]))
        for i in range(gt_.shape[-1])
    ]
    return output


def compute_SE(arr):
    """
    Computes standard error of the mean.
    """
    return np.std(arr) / np.sqrt(len(arr))


def compute_custom_ssim(gt_, pred_, ssim_obj_dict):
    """
    Computes multiscale ssim for each channel.
    Args:
    gt_: ground truth image with shape (N, H, W, C) or List [Hi, Wi, C]
    pred_: predicted image with shape (N, H, W, C)
    range_invariant: whether to use range invariant multiscale ssim
    """
    ms_ssim_values = defaultdict(list)
    cN = gt_[0].shape[-1]
    for i in range(len(gt_)):
        for ch_idx in range(cN):
            tar_tmp = gt_[i][..., ch_idx]
            pred_tmp = pred_[i][..., ch_idx]
            ms_ssim_values[ch_idx].append(
                ssim_obj_dict[ch_idx].score(tar_tmp, pred_tmp)
            )

    output = [
        (np.mean(ms_ssim_values[i]), compute_SE(ms_ssim_values[i])) for i in range(cN)
    ]
    return output




def zero_mean(x):
    return x - torch.mean(x, dim=1, keepdim=True)


def fix_range(gt, x):
    a = torch.sum(gt * x, dim=1, keepdim=True) / (torch.sum(x * x, dim=1, keepdim=True))
    return x * a


def fix(gt, x):
    gt_ = zero_mean(gt)
    return fix_range(gt_, zero_mean(x))


def _PSNR_internal(gt, pred, range_=None):
    if range_ is None:
        range_ = torch.max(gt, dim=1).values - torch.min(gt, dim=1).values

    mse = torch.mean((gt - pred) ** 2, dim=1)
    return 20 * torch.log10(range_ / torch.sqrt(mse))


@allow_numpy
def PSNR(gt, pred, range_=None):
    """
    Compute PSNR.
    Parameters
    ----------
    gt: array
        Ground truth image.
    pred: array
        Predicted image.
    """
    assert len(gt.shape) == 3, "Images must be in shape: (batch,H,W)"

    gt = gt.view(len(gt), -1)
    pred = pred.view(len(gt), -1)
    return _PSNR_internal(gt, pred, range_=range_)


@allow_numpy
def RangeInvariantPsnr(gt, pred):
    """
    NOTE: Works only for grayscale images.
    Adapted from https://github.com/juglab/ScaleInvPSNR/blob/master/psnr.py
    It rescales the prediction to ensure that the prediction has the same range as the ground truth.
    """
    assert len(gt.shape) == 3, "Images must be in shape: (batch,H,W)"
    gt = gt.view(len(gt), -1)
    pred = pred.view(len(gt), -1)
    ra = (torch.max(gt, dim=1).values - torch.min(gt, dim=1).values) / torch.std(
        gt, dim=1
    )
    gt_ = zero_mean(gt) / torch.std(gt, dim=1, keepdim=True)
    return _PSNR_internal(zero_mean(gt_), fix(gt_, pred), ra)


def _avg_psnr(target, prediction, psnr_fn):
    """
    Returns the mean PSNR and the standard error of the mean.
    """
    # multiplication with 1.0 is to ensure that the data is float.
    psnr_arr = [
        psnr_fn(target[i][None] * 1.0, prediction[i][None] * 1.0).item()
        for i in range(len(prediction))
    ]
    mean_psnr = np.mean(psnr_arr)
    std_err_psnr = compute_SE(psnr_arr)
    return round(mean_psnr, 2), round(std_err_psnr, 3)


def avg_range_inv_psnr(target, prediction):
    return _avg_psnr(target, prediction, RangeInvariantPsnr)


def avg_psnr(target, prediction):
    return _avg_psnr(target, prediction, PSNR)


def _get_list_of_images_from_gt_pred(gt, pred, ch_idx):
    """
    Whether you have 2D data or 3D data, this function will return a list of images HixWi.
    """
    gt_list = []
    pred_list = []
    if isinstance(gt, list):
        # assert len(gt[0].shape) == 4, f"expected N x H x W x C, but got {gt[0].shape}"
        for i in range(len(gt)):
            gt_list_tmp, pred_list_tmp = _get_list_of_images_from_gt_pred(
                gt[i], pred[i], ch_idx
            )
            gt_list += gt_list_tmp
            pred_list += pred_list_tmp
    elif isinstance(gt, np.ndarray):
        if len(gt.shape) == 3:
            return [gt[..., ch_idx] * 1.0], [pred[..., ch_idx]]
        else:
            assert (
                gt.shape == pred.shape
            ), f"gt shape: {gt.shape}, pred shape: {pred.shape}"
            for n_idx in range(gt.shape[0]):
                gt_list_tmp, pred_list_tmp = _get_list_of_images_from_gt_pred(
                    gt[n_idx], pred[n_idx], ch_idx
                )
                gt_list += gt_list_tmp
                pred_list += pred_list_tmp
    return gt_list, pred_list


def compute_high_snr_stats(highres_data, pred_unnorm, verbose=True):
    """
    last dimension is the channel dimension
    """
    psnr_list = []
    microssim_list = []
    ms3im_list = []
    ssim_list = []
    msssim_list = []
    for ch_idx in range(highres_data[0].shape[-1]):
        # list of gt and prediction images. This handles both 2D and 3D data. This also handles when individual images are lists.
        gt_ch, pred_ch = _get_list_of_images_from_gt_pred(
            highres_data, pred_unnorm, ch_idx
        )
        # PSNR
        psnr_list.append(avg_range_inv_psnr(gt_ch, pred_ch))

        # MicroSSIM
        microssim_obj = MicroSSIM()
        microssim_obj.fit(gt_ch, pred_ch)
        mssim_scores = [
            microssim_obj.score(gt_ch[i], pred_ch[i]) for i in range(len(gt_ch))
        ]
        microssim_list.append((np.mean(mssim_scores), compute_SE(mssim_scores)))

        # # MicroS3IM
        m3sim_obj = MicroMS3IM()
        m3sim_obj.fit(gt_ch, pred_ch)
        ms3im_scores = [
            m3sim_obj.score(gt_ch[i], pred_ch[i]) for i in range(len(gt_ch))
        ]
        ms3im_list.append((np.mean(ms3im_scores), compute_SE(ms3im_scores)))
        # SSIM
        ssim = [
            structural_similarity(
                gt_ch[i], pred_ch[i], data_range=gt_ch[i].max() - gt_ch[i].min()
            )
            for i in range(len(gt_ch))
        ]
        ssim_list.append((np.mean(ssim), compute_SE(ssim)))
        # MSSSIM
        ms_ssim = []
        for i in range(len(gt_ch)):
            ms_ssim_obj = MultiScaleStructuralSimilarityIndexMeasure(
                data_range=gt_ch[i].max() - gt_ch[i].min()
            )
            ms_ssim.append(
                ms_ssim_obj(
                    torch.Tensor(pred_ch[i][None, None]),
                    torch.Tensor(gt_ch[i][None, None]),
                ).item()
            )
        msssim_list.append((np.mean(ms_ssim), compute_SE(ms_ssim)))
    if verbose:

        def ssim_str(ssim_tmp):
            return f"{np.round(ssim_tmp[0], 3):.3f}+-{np.round(ssim_tmp[1], 3):.3f}"

        def psnr_str(psnr_tmp):
            return f"{np.round(psnr_tmp[0], 2)}+-{np.round(psnr_tmp[1], 3)}"

        print(
            "PSNR on Highres", "\t".join([psnr_str(psnr_tmp) for psnr_tmp in psnr_list])
        )
        print(
            "MicroSSIM on Highres",
            "\t".join([ssim_str(ssim) for ssim in microssim_list]),
        )
        print(
            "MicroS3IM on Highres", "\t".join([ssim_str(ssim) for ssim in ms3im_list])
        )
        print("SSIM on Highres", "\t".join([ssim_str(ssim) for ssim in ssim_list]))
        print("MSSSIM on Highres", "\t".join([ssim_str(ssim) for ssim in msssim_list]))

    return {
        "rangeinvpsnr": psnr_list,
        "microssim": microssim_list,
        "ms3im": ms3im_list,
        "ssim": ssim_list,
        "msssim": msssim_list,
    }
