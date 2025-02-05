# MicroSplit


[![Library](https://img.shields.io/badge/Library-CAREamics-orange)](https://careamics.github.io)
[![License](https://img.shields.io/pypi/l/careamics.svg?color=green)](https://github.com/CAREamics/MicroSplit-reproducibility/blob/main/LICENSE)
[![Image.sc](https://img.shields.io/badge/Got%20a%20question%3F-Image.sc-blue)](https://forum.image.sc/)


## What is MicroSplit

MicroSplit is a deep learning-based computational multiplexing technique that enhances
the imaging of multiple cellular structures within a single fluorescent channel, 
allowing faster imaging and reduced photon exposure.

<p>
    <img src="img/Fig1_a.png" width="800" />
</p>

<p>
    <img src="img/Fig1_b.png" width="800" />
</p>

MicroSplit is based on a hierarchical variational auto-encoder (LVAE) using lateral context.

<p>
    <img src="img/Fig2.png" width="800" />
</p>

MicroSplit is implemented in the [CAREamics library](https://careamics.github.io), and
this repository contains example notebooks and utilities for reproducing the results
of the MicroSplit paper.


## How to use this repository


> [!IMPORTANT]  
> A GPU is necessary for training the models from scratch. For users interested in testing the examples from the paper, our 
notebooks allow loading pre-trained models and running the inference even without GPU access.


### Set up a Python environment

In order to run the examples, you will need to install PyTorch, CAREamics and the utilities in this repository.

1. Create a new environment with the package manager of your choice, we recommand [mamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html), but you can also use [conda](https://docs.anaconda.com/miniconda/) (in which case, substitute `mamba` for `conda` in the following bash commands).
    ```bash
    mamba create -n microsplit python=3.9
    mamba activate microsplit
    ```
    [!TIP]
    If you are on a mac, and wish to make use of mac silicon (M1, M2 and M3 chips), create the environment using the following commands:
    ```bash
    CONDA_SUBDIR=osx-arm64 conda create -n microsplit python=3.9
    conda activate careamics
    conda config --env --set-subdir osx-64
    conda activate microsplit
    ```

2. :warning: Install PyTorch following the instructions on the [official website](https://pytorch.org/get-started/locally/).

3. You can test that you have GPU access by running the following command:
    ```bash
    python -c "import torch; print([torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())])"
    ```
    To confirm that mac silicon is available do:
    ```bash
    python -c "import torch; import platform; print((platform.processor()=='arm' and torch.backends.mps.is_available()))"
    ```

4. Install MicroSplit utilities from this repository by cloning and navigating into it, then by installing all the necessary packages using `pip`.

    ```bash
    git clone https://github.com/CAREamics/MicroSplit-reproducibility.git
    cd MicroSplit-reproducibility
    pip install .
    ```

> [!TIP]  
> If you are on a Windows machine and have trouble running unix-like commands, check out [Git for Windows](https://gitforwindows.org/). This tool installs Git Bash, a terminal that you can use to run the commands above.


### Clone the repository to access the examples

5. You can now open the notebooks in `jupyter` by running the following command and navigating to the example folder:

    ```bash
    jupyter notebook
    ```

> [!NOTE]  
> The Jupyter notebooks in each example are numbered by their order in the MicroSplit pipeline:
> - 00: Create the noise models for the dataset
> - 01: Train the MicroSplit model
> - 02: Apply MicroSplit to data
> - 03: Calibrate the MicroSplit errors
>
> The notebooks are designed to be run in order, but we designed them so that each notebook, except the calibration, has entry points using pre-trained models.

## Useful links

- [CAREamics documentation](https://careamics.github.io)
- (soon) [MicroSplit algorithm summary]()
- (soon) [Noise models summary]()


## Cite MicroSplit

<!--- Add citation --->

(soon)

## Links to all dataseets used in the manuscript

<!--- Add links to dataset zip files on download.fht.org --->

(soon)

## License

This project is licensed under BSD-3-Clause License - see the [LICENSE](LICENSE) for details.
