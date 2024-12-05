import random

import numpy as np
from pandas import read_csv
import torch
import matplotlib.pyplot as plt


def fix_seeds(seed: int = 0) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_ignored_pixels(pred: torch.Tensor) -> int:
    """Get the number of ignored pixels in the predictions.

    Some of the pixels present in the last few rows and columns of predicted images
    should be discarded. Indeed, the prediction there is always zero, since they don't
    come in batches as not multiples of the tile size.

    To spot these areas, we analyze the pixel-wise std of a predicted image. When that
    is zero, we know that the last few rows and columns are all zero.

    NOTE: this function is just to understand the number of ignored pixels.
    The function to discard these pixels is `ignore_pixels`.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted image. Shape is (H, W, C).

    Returns
    -------
    int
        Number of ignored pixels.
    """
    ignored_pixels = 1
    while pred[-ignored_pixels:, -ignored_pixels:, ...].std() == 0:
        ignored_pixels += 1
    ignored_pixels -= 1
    print(f"In {pred.shape}, last {ignored_pixels} many rows and columns are zero.")
    return ignored_pixels


def plot_probability_distribution(noise_model, signalBinIndex, histogram, channel):
    # TODO add typing
    """Plots probability distribution P(x|s) for a certain ground truth signal.

    Predictions from both Histogram and GMM-based Noise models are displayed for comparison.

    Parameters
    ----------
    signalBinIndex: int
        index of signal bin. Values go from 0 to number of bins (`n_bin`).
    histogramNoiseModel: Histogram based noise model
    gaussianMixtureNoiseModel: GaussianMixtureNoiseModel
        Object containing trained parameters.
    device: GPU device
    """

    n_bin = 100  # TODO clarify this and signalBinIndex
    histBinSize = (
        noise_model.max_signal.item() - noise_model.min_signal.item()
    ) / n_bin
    querySignal = (
        signalBinIndex
        / float(n_bin)
        * (noise_model.max_signal - noise_model.min_signal)
        + noise_model.min_signal
    )
    querySignal += histBinSize / 2

    queryObservations = torch.arange(
        noise_model.min_signal.item(), noise_model.max_signal.item(), histBinSize
    )
    queryObservations += histBinSize / 2
    # TODO this is ugly, refactor
    noise_model.mode = "inference"
    noise_model.min_signal = torch.tensor(
        noise_model.min_signal, device=noise_model.device
    )
    noise_model.max_signal = torch.tensor(
        noise_model.max_signal, device=noise_model.device
    )
    # noise_model.tol = torch.tensor(noise_model.tol, device=noise_model.device)
    pTorch = noise_model.likelihood(
        observations=queryObservations, signals=torch.tensor(querySignal)
    )
    pNumpy = pTorch.cpu().detach().numpy()

    plt.figure(figsize=(12, 5))
    plt.suptitle(f"Noise model for channel {channel}")
    plt.subplot(1, 2, 1)
    plt.xlabel("Observation Bin")
    plt.ylabel("Signal Bin")
    plt.imshow(histogram**0.25, cmap="gray")
    plt.axhline(y=signalBinIndex + 0.5, linewidth=5, color="blue", alpha=0.5)
    plt.subplot(1, 2, 2)

    # histobs_repeated = np.repeat(histobs, 2)
    # queryObservations_repeated = np.repeat(queryObservations_numpy, 2)

    plt.plot(
        queryObservations,
        pNumpy,
        label="GMM : " + " signal = " + str(np.round(querySignal, 2)),
        marker=".",
        color="red",
        linewidth=2,
    )
    plt.xlabel("Observations (x) for signal s = " + str(querySignal))
    plt.ylabel("Probability Density")
    plt.title("Probability Distribution P(x|s) at signal =" + str(querySignal))

    plt.legend()
    return {
        "gmm": {"x": queryObservations, "p": pNumpy},
    }


def plot_training_metrics(file_path: str):
    csv_file = read_csv(file_path)
    csv_file.plot(x="epoch", y=["val_loss"]) # TODO add more metrics to plot
