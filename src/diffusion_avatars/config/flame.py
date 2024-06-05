import argparse
from dataclasses import dataclass

from elias.config import Config


@dataclass(unsafe_hash=True)
class FlameConfig(Config):
    shape_params: int = 100
    expression_params: int = 50
    pose_params: int = 6
    use_face_contour: bool = True  # If true apply the landmark loss on also on the face contour
    use_3D_translation: bool = True  # If true apply the landmark loss on also on the face contour
    optimize_eyeballpose: bool = True  # If true optimize for the eyeball pose
    optimize_neckpose: bool = True  # If true optimize for the neck pose
    num_worker: int = 4
    batch_size: int = 8
    ring_margin: float = 0.5
    ring_loss_weight: float = 1.0
    flame_version: str = 'generic_model.pkl'


parser = argparse.ArgumentParser(description='FLAME model')

parser.add_argument(
    '--flame_model_path',
    type=str,
    default='./model/generic_model.pkl',
    help='flame model path'
)

parser.add_argument(
    '--static_landmark_embedding_path',
    type=str,
    default='./model/flame_static_embedding.pkl',
    help='Static landmark embeddings path for FLAME'
)

parser.add_argument(
    '--dynamic_landmark_embedding_path',
    type=str,
    default='./model/flame_dynamic_embedding.npy',
    help='Dynamic contour embedding path for FLAME'
)

# FLAME hyper-parameters

parser.add_argument(
    '--shape_params',
    type=int,
    default=100,
    help='the number of shape parameters'
)

parser.add_argument(
    '--expression_params',
    type=int,
    default=50,
    help='the number of expression parameters'
)

parser.add_argument(
    '--pose_params',
    type=int,
    default=6,
    help='the number of pose parameters'
)

# Training hyper-parameters

parser.add_argument(
    '--use_face_contour',
    default=True,
    type=bool,
    help='If true apply the landmark loss on also on the face contour.'
)

parser.add_argument(
    '--use_3D_translation',
    default=True,  # Flase for RingNet project
    type=bool,
    help='If true apply the landmark loss on also on the face contour.'
)

parser.add_argument(
    '--optimize_eyeballpose',
    default=True,  # False for For RingNet project
    type=bool,
    help='If true optimize for the eyeball pose.'
)

parser.add_argument(
    '--optimize_neckpose',
    default=True,  # False For RingNet project
    type=bool,
    help='If true optimize for the neck pose.'
)

parser.add_argument(
    '--num_worker',
    type=int,
    default=4,
    help='pytorch number worker.'
)

parser.add_argument(
    '--batch_size',
    type=int,
    default=8,
    help='Training batch size.'
)

parser.add_argument(
    '--ring_margin',
    type=float,
    default=0.5,
    help='ring margin.'
)

parser.add_argument(
    '--ring_loss_weight',
    type=float,
    default=1.0,
    help='weight on ring loss.'
)


def get_config():
    config = parser.parse_args()
    return config
