from enum import Enum
from typing import List, Optional
from omegaconf import MISSING
from dataclasses import dataclass, field
from tensorflow_caney.model.config.cnn_config import Config


@dataclass
class LandUseConfig(Config):

    data_regex: Optional[list] = None

    segmentation_num_clusters: int = 80

    segmentation_min_n_pxls: int = 8000
