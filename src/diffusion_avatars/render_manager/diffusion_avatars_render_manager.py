from dataclasses import dataclass
from typing import List, Optional, Literal

import mediapy
import numpy as np
from elias.config import Config
from elias.folder import Folder
from elias.util import ensure_directory_exists, save_img, load_img, save_json, load_json

from diffusion_avatars.env import DIFFUSION_AVATARS_RENDERS_PATH

# from diff_vp.renderer.provider.expression_animation_provider import ExpressionAnimationManager

# 'camera_moveX', 'camera_shake', 'camera_interpolation', 'camera'

TrajectoryType = Literal[
    "camera_moveX",  # A circular trajectory with a train sequence replayed but FLAME rendered from novel viewpoints
    "camera_shake",  # camera shakes left and right
    "camera",  # Render frome one of the 16 dataset viewpoints
]


@dataclass
class DiffusionAvatarsTrajectory(Config):
    cam_2_world_poses: List[np.ndarray]
    intrinsics: np.ndarray
    sequence: str
    timesteps: List[int]


class DiffusionAvatarsRenderManager:

    def __init__(self,
                 run_name: str,
                 plot_type: TrajectoryType,
                 checkpoint: Optional[int] = None,
                 disable_previous_output: bool = False,
                 disable_neural_texture: bool = False,
                 n_inference_steps: int = 20,
                 use_ddpm_scheduler: bool = False,
                 use_no_variance_scheduler: bool = False,
                 use_unseen_persons: bool = False,
                 use_consistency_decoder: bool = False,
                 enable_extended_temporal_attention: bool = False,
                 smooth_expression_condition: bool = False,
                 resolution: Optional[int] = None,
                 camera_distance: float = 1,
                 source_actor: Optional[int] = None,
                 source_sequence: Optional[str] = None,
                 source_animation: Optional[str] = None,
                 prompt: Optional[str] = None,
                 cfg_weight: float = 1,
                 scheduler: Optional['SchedulerType'] = None,
                 shake: float = 0.3,
                 move_y: float = None,
                 look_y: Optional[float] = None,
                 curve_z: float = None,
                 test_sequence: Optional[int] = None,
                 frequency: int = 40):

        parts = run_name.split('-')
        prefix = '-'.join(parts[:-1])
        # model_folder = find_model_folder(run_name)
        # prefix = model_folder.get_prefix()
        self._run_name = run_name
        self._plot_type = plot_type
        self._checkpoint = checkpoint
        self._location = f"{DIFFUSION_AVATARS_RENDERS_PATH}/{prefix}"
        self._disable_previous_output = disable_previous_output
        self._disable_neural_texture = disable_neural_texture
        self._n_inference_steps = n_inference_steps
        self._use_ddpm_scheduler = use_ddpm_scheduler
        self._use_no_variance_scheduler = use_no_variance_scheduler
        self._use_unseen_persons = use_unseen_persons
        self._use_consistency_decoder = use_consistency_decoder
        self._enable_extended_temporal_attention = enable_extended_temporal_attention
        self._smooth_expression_condition = smooth_expression_condition
        self._resolution = resolution
        self._camera_distance = camera_distance
        self._source_actor = source_actor
        self._source_sequence = source_sequence
        self._source_animation = source_animation
        self._prompt = prompt
        self._cfg_weight = cfg_weight
        self._scheduler = scheduler
        self._shake = shake
        self._curve_z = curve_z
        self._move_y = move_y
        self._look_y = look_y
        self._frequency = frequency
        self._test_sequence = test_sequence

        if checkpoint < 0:
            self._checkpoint = self.get_checkpoint_ids()[checkpoint]

    def save_video(self, frames: List[np.ndarray], name: str, fps: int = 24, use_run_folder: bool = False):
        folder = f"{self._location}/{self._plot_type}"

        if use_run_folder:

            if self._source_actor is not None:
                run_folder = f"{folder}/{self._run_name}/from_{self._source_actor}_{self._source_sequence}"
            elif self._source_animation is not None:
                run_folder = f"{folder}/{self._run_name}/anim_{self._source_animation}"
            else:
                run_folder = f"{folder}/{self._run_name}"
            ensure_directory_exists(run_folder)
            mediapy.write_video(f"{run_folder}/{name}.mp4", frames, fps=fps)
        else:
            ensure_directory_exists(folder)
            mediapy.write_video(f"{folder}/{self._run_name}_{name}.mp4", frames, fps=fps)

    def save_frames(self, frames: List[np.ndarray], name: str):
        folder = f"{self._location}/{self._plot_type}"
        if self._source_actor is not None:
            run_folder = f"{folder}/{self._run_name}/from_{self._source_actor}_{self._source_sequence}/frames_{name}"
        elif self._source_animation is not None:
            run_folder = f"{folder}/{self._run_name}/anim_{self._source_animation}"
        else:
            run_folder = f"{folder}/{self._run_name}/frames_{name}"

        expression_animation_manager = None
        if self._source_animation is not None:
            expression_animation_manager = ExpressionAnimationManager(self._source_animation, skip=2)

        if expression_animation_manager is not None and expression_animation_manager.is_matrix():
            timestep_matrix = expression_animation_manager.get_timestep_matrix()
            for timestep, frame in zip(expression_animation_manager.get_timesteps(), frames):
                step_1, step_2 = timestep_matrix[timestep]
                save_img(frame, f"{run_folder}/frame_{step_1:05d}_{step_2:05d}.jpg")
        else:
            for i, frame in enumerate(frames):
                save_img(frame, f"{run_folder}/frame_{i:05d}.png")

    def get_generations_video_path(self, fix_latents: bool = False,
                                   fix_timestep: bool = False,
                                   seed: Optional[int] = None,
                                   prompt: Optional[str] = None,
                                   person: Optional[int] = None) -> str:

        name = self.get_generations_name(fix_latents=fix_latents, fix_timestep=fix_timestep, seed=seed, prompt=prompt, person=person)
        folder = f"{self._location}/{self._plot_type}"

        if self._source_actor is not None:
            run_folder = f"{folder}/{self._run_name}/from_{self._source_actor}_{self._source_sequence}"
        elif self._source_animation is not None:
            run_folder = f"{folder}/{self._run_name}/anim_{self._source_animation}"
        else:
            run_folder = f"{folder}/{self._run_name}"
        ensure_directory_exists(run_folder)

        return f"{run_folder}/{name}.mp4"

    def get_conditions_video_path(self,
                                  rendering_name: 'RenderingName',
                                  fix_timestep: bool = False,
                                  person: Optional[int] = None):
        name = self.get_conditions_name(rendering_name, fix_timestep=fix_timestep, person=person)
        folder = f"{self._location}/{self._plot_type}"

        if self._source_actor is not None:
            run_folder = f"{folder}/{self._run_name}/from_{self._source_actor}_{self._source_sequence}"
        elif self._source_animation is not None:
            run_folder = f"{folder}/{self._run_name}/anim_{self._source_animation}"
        else:
            run_folder = f"{folder}/{self._run_name}"
        ensure_directory_exists(run_folder)

        return f"{run_folder}/{name}.mp4"

    def save_conditions_video(self,
                              frames: List[np.ndarray],
                              rendering_name: 'RenderingName',
                              fps: int = 24,
                              fix_timestep: bool = False,
                              person: Optional[int] = None):
        name = self.get_conditions_name(rendering_name, fix_timestep=fix_timestep, person=person)

        self.save_video(frames, name, fps=fps, use_run_folder=True)

    def save_generations_video(self,
                               frames: List[np.ndarray],
                               fps: int = 24,
                               fix_latents: bool = False,
                               fix_timestep: bool = False,
                               seed: Optional[int] = None,
                               prompt: Optional[str] = None,
                               person: Optional[int] = None):
        name = self.get_generations_name(fix_latents=fix_latents, fix_timestep=fix_timestep, seed=seed, prompt=prompt, person=person)

        self.save_video(frames, name, fps=fps, use_run_folder=False)
        self.save_video(frames, name, fps=fps, use_run_folder=True)

    def load_conditions_video(self,
                              rendering_name: 'RenderingName',
                              fix_timestep: bool = False,
                              person: Optional[int] = None) -> np.ndarray:
        video_path = self.get_conditions_video_path(rendering_name, fix_timestep=fix_timestep, person=person)
        frames = mediapy.read_video(video_path)
        return np.asarray(frames)

    def get_generations_name(self, fix_latents: bool = False,
                             fix_timestep: bool = False,
                             seed: Optional[int] = None,
                             prompt: Optional[str] = None,
                             person: Optional[int] = None) -> str:
        name = "generations"
        if fix_latents:
            name += "_fix-latents"
        if fix_timestep:
            name += "_fix-timestep"
        if seed is not None:
            name += f"_seed-{seed}"
        if prompt is not None:
            name += f"_prompt=({prompt.replace(' ', '-')})"
        if self._use_unseen_persons:
            name += f"_unseen-persons"
        if person is not None:
            name += f"_person-{person}"
        if self._n_inference_steps != 20:
            name += f"_inference-steps-{self._n_inference_steps}"
        if self._use_ddpm_scheduler:
            name += "_ddpm-scheduler"
        if self._use_no_variance_scheduler:
            name += "_no-variance-ddpm-scheduler"
        if self._scheduler is not None:
            name += f"_scheduler-{self._scheduler}"
        if self._use_consistency_decoder:
            name += "_consistency-decoder"
        if self._enable_extended_temporal_attention:
            name += "_extended-temp-att"
        if self._smooth_expression_condition:
            name += "_smooth-expr-cond"
        if self._resolution is not None:
            name += f"_resolution-{self._resolution}"
        if self._test_sequence is not None:
            name += f"_test-sequence-{self._test_sequence}"
        if self._camera_distance != 1 and self._plot_type != 'camera':
            name += f"_distance-{self._camera_distance}"
        if self._shake != 0.3 and self._plot_type == 'camera_shake':
            name += f"_shake-{self._shake}"
        if self._move_y is not None and self._plot_type in {'camera_shake', 'camera'}:
            name += f"_move-y-{self._move_y}"
        if self._look_y is not None and self._plot_type == 'camera_shake':
            name += f"_look-y-{self._look_y}"
        if self._curve_z is not None and self._plot_type == 'camera_shake':
            name += f"_curve_z-{self._curve_z}"
        if self._frequency != 40 and self._plot_type == 'camera_shake':
            name += f"_freq-{self._frequency}"
        if self._source_actor is not None:
            name += f"_from-{self._source_actor}_{self._source_sequence}"
        if self._source_animation is not None:
            name += f"_anim-{self._source_animation}"
        if self._prompt is not None:
            name += f"_prompt-{self._prompt.replace(' ', '-')}_cfg-{self._cfg_weight}"

        if self._disable_previous_output:
            name += f"_disable-previous-output"
        if self._disable_neural_texture:
            name += f"_disable-neural-texture"
        if self._checkpoint is not None:
            name += f"_ckpt-{self._checkpoint:09d}"

        return name

    def get_conditions_name(self,
                            rendering_name: 'RenderingName',
                            fix_timestep: bool = False,
                            person: Optional[int] = None):
        name = f"condition-{rendering_name}"
        if fix_timestep:
            name += "_fix-latents"
        if self._use_unseen_persons:
            name += f"_unseen-persons"
        if person is not None:
            name += f"_person-{person}"
        if self._resolution is not None:
            name += f"_resolution-{self._resolution}"
        if self._test_sequence is not None:
            name += f"_test-sequence-{self._test_sequence}"
        if self._camera_distance != 1:
            name += f"_distance-{self._camera_distance}"
        if self._shake != 0.3 and self._plot_type == 'camera_shake':
            name += f"_shake-{self._shake}"
        if self._move_y is not None and self._plot_type in {'camera_shake', 'camera'}:
            name += f"_move-y-{self._move_y}"
        if self._look_y is not None and self._plot_type == 'camera_shake':
            name += f"_look-y-{self._look_y}"
        if self._curve_z is not None and self._plot_type == 'camera_shake':
            name += f"_curve_z-{self._curve_z}"
        if self._frequency != 40 and self._plot_type == 'camera_shake':
            name += f"_freq-{self._frequency}"

        if self._disable_previous_output:
            name += f"_disable-previous-output"
        if self._disable_neural_texture and rendering_name == 'neural_textured_rendering':
            name += f"_disable-neural-texture"
        if self._checkpoint is not None:
            name += f"_ckpt-{self._checkpoint:09d}"

        return name

    def save_generation_frames(self, frames: List[np.ndarray]):
        name = "generations"
        if self._test_sequence is not None:
            name += f"_test-sequence-{self._test_sequence}"
        if self._camera_distance != 1:
            name += f"_distance-{self._camera_distance}"
        if self._shake != 0.3 and self._plot_type == 'camera_shake':
            name += f"_shake-{self._shake}"
        if self._move_y is not None and self._plot_type in {'camera_shake', 'camera'}:
            name += f"_move-y-{self._move_y}"
        if self._look_y is not None and self._plot_type == 'camera_shake':
            name += f"_look-y-{self._look_y}"
        if self._curve_z is not None and self._plot_type == 'camera_shake':
            name += f"_curve_z-{self._curve_z}"
        if self._frequency != 40 and self._plot_type == 'camera_shake':
            name += f"_freq-{self._frequency}"
        if self._checkpoint is not None:
            name += f"_ckpt-{self._checkpoint:09d}"

        self.save_frames(frames, name)

    def load_generation_frame(self, frame_id: int) -> np.ndarray:
        frame = load_img(self.get_generation_path(frame_id))
        return frame

    def get_generation_timesteps(self) -> List[int]:
        run_folder = self.get_generation_folder()

        timesteps = Folder(run_folder).list_file_numbering('frame_$.png', return_only_numbering=True)
        return timesteps

    def get_generation_path(self, frame_id: int) -> str:
        run_folder = self.get_generation_folder()
        return f"{run_folder}/frame_{frame_id:05d}.png"

    def get_generation_folder(self) -> str:
        name = "generations"
        if self._test_sequence is not None:
            name += f"_test-sequence-{self._test_sequence}"
        if self._camera_distance != 1:
            name += f"_distance-{self._camera_distance}"
        if self._shake != 0.3 and self._plot_type == 'camera_shake':
            name += f"_shake-{self._shake}"
        if self._move_y is not None and self._plot_type in {'camera_shake', 'camera'}:
            name += f"_move-y-{self._move_y}"
        if self._look_y is not None and self._plot_type == 'camera_shake':
            name += f"_look-y-{self._look_y}"
        if self._curve_z is not None and self._plot_type == 'camera_shake':
            name += f"_curve_z-{self._curve_z}"
        if self._frequency != 40 and self._plot_type == 'camera_shake':
            name += f"_freq-{self._frequency}"
        if self._checkpoint is not None:
            name += f"_ckpt-{self._checkpoint:09d}"

        folder = f"{self._location}/{self._plot_type}"
        if self._source_actor is not None:
            run_folder = f"{folder}/{self._run_name}/from_{self._source_actor}_{self._source_sequence}/frames_{name}"
        elif self._source_animation is not None:
            run_folder = f"{folder}/{self._run_name}/anim_{self._source_animation}"
        else:
            run_folder = f"{folder}/{self._run_name}/frames_{name}"

        return run_folder

    def get_nersemble_folder(self, checkpoint: int) -> str:
        folder = f"{self._location}/{self._plot_type}"
        if self._source_actor is not None:
            run_folder = f"{folder}/{self._run_name}/from_{self._source_actor}_{self._source_sequence}"
        elif self._source_animation is not None:
            run_folder = f"{folder}/{self._run_name}/anim_{self._source_animation}"
        else:
            run_folder = f"{folder}/{self._run_name}"

        return f"{run_folder}/nersemble-{checkpoint}"

    def get_checkpoint_ids(self) -> List[int]:
        folder = f"{self._location}/{self._plot_type}"
        if self._source_actor is not None:
            run_folder = f"{folder}/{self._run_name}/from_{self._source_actor}_{self._source_sequence}"
        elif self._source_animation is not None:
            run_folder = f"{folder}/{self._run_name}/anim_{self._source_animation}"
        else:
            run_folder = f"{folder}/{self._run_name}"

        if self._source_animation is None:
            checkpoint_ids = Folder(run_folder).list_file_numbering('frames_generations*_ckpt-$', return_only_numbering=True)
        else:
            checkpoint_ids = Folder(run_folder).list_file_numbering('generations*_ckpt-$.mp4', return_only_numbering=True)
        return checkpoint_ids

    def save_figure(self,
                    figure: np.ndarray,
                    prompt: Optional[str] = None):
        name = self._plot_type

        if prompt is not None:
            name += f"_prompt=({prompt.replace(' ', '-')})"
        if self._checkpoint is not None:
            name += f"_ckpt-{self._checkpoint:09d}"
        if self._disable_previous_output:
            name += f"_disable-previous-output"
        if self._n_inference_steps != 20:
            name += f"_inference-steps-{self._n_inference_steps}"
        if self._use_ddpm_scheduler:
            name += "_ddpm-scheduler"
        if self._use_no_variance_scheduler:
            name += "_no-variance-ddpm-scheduler"
        if self._use_unseen_persons:
            name += f"_unseen-persons"

        folder = f"{self._location}/{self._plot_type}"
        ensure_directory_exists(folder)
        save_img(figure, f"{folder}/{self._run_name}_{name}.jpg")
