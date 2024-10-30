from typing import Literal, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field


class SplittingParameters(BaseModel):
    """Base configuration for the parameters used in splitting algorithms."""
    
    config_dict = ConfigDict(validate_assignment=True, validate_default=True)
    
    # --- Custom parameters
    img_size: Sequence[int]
    """Spatial size of the input image."""
    
    target_channels: int = Field(..., ge=1)
    """Number of channels in the target image."""
    
    multiscale_count: int = Field(..., ge=1)
    """The number of LC inputs plus one (the actual input)."""
    
    predict_logvar: Optional[Literal["pixelwise"]]
    """Whether to compute also the log-variance as LVAE output."""
    
    loss_type: Optional[Literal["musplit", "denoisplit", "denoisplit_musplit"]]
    """The type of reconstruction loss (i.e., likelihood) to use."""
    
    nm_paths: Optional[Sequence[str]] = Field(..., min_items=1)
    """The paths to the pre-trained noise models for the different channels."""
    # ---

    # --- Training Parameters
    batch_size: int = Field(32, ge=1, le=256)
    """The batch size for training."""
    
    lr: float = Field(1e-3, ge=1e-10)
    """The learning rate for training."""
    
    lr_scheduler_patience: int = Field(30, ge=5)
    """The patience for the learning rate scheduler."""
    
    earlystop_patience: int = Field(200, ge=10)
    """The patience for the learning rate scheduler."""
    
    max_epochs: int = Field(400, ge=1)
    """The maximum number of epochs to train for."""
    
    num_workers: int = 4
    """The number of workers to use for data loading."""
    # ---