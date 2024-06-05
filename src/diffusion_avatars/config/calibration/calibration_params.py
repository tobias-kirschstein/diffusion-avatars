from dataclasses import dataclass
from enum import auto
from typing import List, Optional, Dict

import numpy as np
from dreifus.matrix import Pose, Intrinsics
from dreifus.vector import Vec3
from elias.config import Config, StringEnum


class CornerDisplacementMode(StringEnum):
    NONE = auto()
    GLOBAL = auto()
    EVERY_POSITION = auto()


@dataclass
class IntrinsicParams(Config):
    cx: float
    cy: float
    fx: float
    fy: float = None


@dataclass
class MultiPosCalibrationParams(Config):
    # fx: float = None  # deprecated
    # cx: float = None  # deprecated
    # cy: float = None  # deprecated
    ts: np.ndarray = None  # [ I x 3 ]
    rs: np.ndarray = None  # [ I x 3 ]
    checkerboard_poses: List[np.ndarray] = None  # 2 (6,)
    corner_displacements: Optional[np.ndarray] = None  # Either #C x 3 or P x C x 3
    # fy: Optional[float] = None  # If fy is not specified, assume fx = fy
    intrinsics: List[IntrinsicParams] = None

    @classmethod
    def _backward_compatibility(cls, loaded_config: Dict):
        super()._backward_compatibility(loaded_config)

        if 'fx' in loaded_config:
            # Older calibration params only had global intrinsics that were the same for all cameras
            # Newer runs use a list of Intrinsic Params instead
            # Simply create a list containing only one set of intrinsic params in this case
            fx = loaded_config['fx']
            cx = loaded_config['cx']
            cy = loaded_config['cy']
            fy = loaded_config['fy'] if 'fy' in loaded_config else fx

            loaded_config['intrinsics'] = [IntrinsicParams(cx, cy, fx, fy)]

            del loaded_config['fx']
            del loaded_config['cx']
            del loaded_config['cy']
            if 'fy' in loaded_config:
                del loaded_config['fy']

    @property
    def fx(self) -> float:
        return self.intrinsics[0].fx

    @property
    def fy(self) -> float:
        return self.intrinsics[0].fy

    @property
    def cx(self) -> float:
        return self.intrinsics[0].cx

    @property
    def cy(self) -> float:
        return self.intrinsics[0].cy

    def get_intrinsics(self, cam_id: int = 0) -> Intrinsics:
        intrinsic_params = self.intrinsics[cam_id]
        return Intrinsics(matrix_or_fx=intrinsic_params.fx,
                          fy=intrinsic_params.fy,
                          cx=intrinsic_params.cx,
                          cy=intrinsic_params.cy)

    def get_intrinsic_params(self, cam_id: int = 0) -> IntrinsicParams:
        return self.intrinsics[cam_id]

    def get_all_intrinsic_params(self) -> List[IntrinsicParams]:
        return self.intrinsics

    def get_all_intrinsics(self) -> List[Intrinsics]:
        n_poses = self.get_n_poses()
        if n_poses > len(self.intrinsics):
            return [self.get_intrinsics(0) for _ in range(n_poses)]  # All cameras have the same Intrinsics
        else:
            return [self.get_intrinsics(cam_id) for cam_id in range(n_poses)]

    def get_poses(self) -> List[Pose]:
        return [self.get_pose(cam_id) for cam_id in range(self.get_n_poses())]

    def get_pose(self, cam_id: int) -> Pose:
        """
        Gives `ref_checkerboard_to_cam`, i.e., the reference checkerboard (usually the first position) as seen
        from cam `i_img`.

        Parameters
        ----------
            cam_id: index of the cam

        Returns
        -------
            ref_checkerboard_to_cam
        """

        return Pose.from_rodriguez(self.rs[cam_id], self.ts[cam_id])

    def get_translation(self, cam_id: int) -> Vec3:
        """
        Gives the translation part of a calibrated camera's world_to_cam matrix.

        Parameters
        ----------
            cam_id: For which camera the translation should be obtained
        """
        return Vec3(self.ts[cam_id])

    def get_n_poses(self) -> int:
        return len(self.rs)

    def get_checkerboard_pose(self, pos: int) -> Pose:
        if pos == 0:
            # First position does not need a transformation because it is the reference position
            return Pose()
        else:
            pos -= 1
            return Pose.from_rodriguez(self.checkerboard_poses[pos][3:6], self.checkerboard_poses[pos][:3])

    def get_corner_displacements(self, pos: Optional[int] = None) -> Optional[np.ndarray]:
        if pos is None or self.corner_displacements is None or len(self.corner_displacements.shape) == 2:
            return self.corner_displacements
        else:
            assert len(self.corner_displacements.shape) == 3
            return self.corner_displacements[pos]

    def get_corner_displacement_mode(self) -> CornerDisplacementMode:
        if self.corner_displacements is None:
            return CornerDisplacementMode.NONE
        elif len(self.corner_displacements.shape) == 2:
            return CornerDisplacementMode.GLOBAL
        elif len(self.corner_displacements.shape) == 3:
            return CornerDisplacementMode.EVERY_POSITION
        raise AssertionError("Illegal state of corner_displacements")

    def get_n_total_valid_images(self):
        return len(self.ts)
