from dataclasses import dataclass
from enum import auto
from typing import List, Optional

import numpy as np
from elias.config import Config, StringEnum

from dreifus.graphics import Dimensions

from diffusion_avatars.config.calibration.calibration_params import CornerDisplacementMode, MultiPosCalibrationParams


class OptimizationMode(StringEnum):
    SCIPY = auto()
    TORCH = auto()
    GD = auto()
    CERES = auto()




@dataclass
class CalibrationRunConfig(Config):
    dataset_version: str
    positions: List[int]
    lambda_corner_displacements: Optional[float] = 1000
    lambda_principal_point: Optional[float] = 100
    lambda_focal_length: Optional[float] = 0
    corner_displacement_mode: Optional[CornerDisplacementMode] = None
    optimization_mode: OptimizationMode = OptimizationMode.SCIPY
    use_double: bool = False
    serials: List[str] = None
    image_size: Dimensions = None
    optimizer_per_camera_intrinsics: bool = False


@dataclass
class CalibrationResult(Config):
    params_result: MultiPosCalibrationParams
    detected_corners: List[List[Optional[np.ndarray]]]
    runtime: float
    cost_history: Optional[List[float]]
    n_residuals: Optional[int]


@dataclass
class CalibrationEvaluationResult(Config):
    location_error: float
    rotation_error: float
    reprojection_error: float
    focal_length_error: float
    principal_point_error: float
    corner_displacements_error: Optional[float]
    cam_coverage: int
    runtime: float
