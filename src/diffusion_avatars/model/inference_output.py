from dataclasses import dataclass
from typing import Dict, Optional, List, Literal

import numpy as np
from diffusion_avatars.data_manager.rendering_data_manager import RenderingName

SchedulerType = Literal['unipc', 'ddim', 'ddpm', 'no-variance-ddpm', 'linear-ddpm']

@dataclass
class DiffusionAvatarsInferenceOutput:
    prediction: np.ndarray
    conditions: Dict[RenderingName, np.ndarray]
    target_image: Optional[np.ndarray] = None
    latent_noise: Optional[np.ndarray] = None
    neural_textured_rendering: Optional[np.ndarray] = None
    ambient_textured_rendering: Optional[np.ndarray] = None
    previous_outputs: Optional[List[np.ndarray]] = None
    prompt: str = ""


