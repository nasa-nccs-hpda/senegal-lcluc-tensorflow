from enum import Enum
from typing import List, Optional
from omegaconf import MISSING
from dataclasses import dataclass, field
from tensorflow_caney.model.config.cnn_config import Config


@dataclass
class LandCoverConfig(Config):

    test_classes: List[str] = field(
        default_factory=lambda: ['other', 'tree', 'crop']
    )

    test_colors: List[str] = field(
        default_factory=lambda: ['brown', 'forestgreen', 'orange']
    )

    """
    # List of strings to support the modification of labels
    modify_test_labels: Optional[List[str]] = field(
        default_factory=lambda: [
              - "x == 0": 8
                - "x == 1": 9
                - "x == 4": 7
                - "x == 3": 0
                - "x == 2": 0
                - "x == 8": 1
                - "x == 9": 2
                - "x == 7": 3
        ]
    )
    """

    validation_buffer_kernel: Optional[float] = 3.7
    validation_cap_style: Optional[int] = 3
    validation_epsg: Optional[str] = "EPSG:32628"
