from pathlib import Path

from environs import Env

env = Env(expand_vars=True)
env_file_path = Path(f"{Path.home()}/.config/diffusion-avatars/.env")
if env_file_path.exists():
    env.read_env(str(env_file_path), recurse=False)

with env.prefixed("DIFFUSION_AVATARS_"):
    DIFFUSION_AVATARS_DATA_PATH = env("DATA_PATH")
    DIFFUSION_AVATARS_MODELS_PATH = env("MODELS_PATH")
    DIFFUSION_AVATARS_RENDERS_PATH = env("RENDERS_PATH")

    DIFFUSION_AVATARS_FLAME_MODEL_PATH = env("FLAME_MODEL_PATH", f"<<<Define DIFFUSION_AVATARS_FLAME_MODEL_PATH in {env_file_path}>>>")


REPO_ROOT_DIR = f"{Path(__file__).parent.resolve()}/../.."