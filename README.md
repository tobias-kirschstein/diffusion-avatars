# DiffusionAvatars: Deferred Diffusion for High-fidelity 3D Head Avatars

[Paper](https://arxiv.org/pdf/2311.18635.pdf) | [Video](https://youtu.be/nSjDiiTnp2E) | [Project Page](https://tobias-kirschstein.github.io/diffusion-avatars/)

![](static/diffusion_avatars_teaser.gif)

[Tobias Kirschstein](https://tobias-kirschstein.github.io/), [Simon Giebenhain](https://simongiebenhain.github.io/)
and [Matthias Nießner](https://niessnerlab.org/)  
**CVPR 2024**

# 1. Installation

### 1.1. Dependencies

1. Setup environment
   ```
   conda env create -f environment.yml
   conda activate diffusion-avatars
   ```
   which creates a new conda environment `diffusion-avatars` (Installation may take a while).

2. Install the `diffusion_avatars` package itself by running
   ```shell
   pip install -e .
   ```
   inside the cloned repository folder.

3. *[Optional Linux]* Update `LD_LIBRARY_PATH` for nvidffrast
   ```shell
   ln -s "$CONDA_PREFIX/lib" "$CONDA_PREFIX/lib64"
   ```
   Solves the issue `/usr/bin/ld: cannot find -lcudart`

4. *[Optional Windows]* Update `CUDA_HOME` for nvidiffrast
   ```shell
   conda env config vars set CUDA_HOME=$Env:CONDA_PREFIX
   conda activate base
   conda activate diffusion-avatars
   ```
   Solves the issue when `nvidffrast` wants to use a globally installed CUDA toolkit instead of the one from the environment.

### 1.2. Environment Paths

All paths to data / models / renderings are defined by _environment variables_.  
Please create a file in your home directory in `~/.config/diffusion-avatars/.env` with the following content:

```shell
DIFFUSION_AVATARS_DATA_PATH="..."
DIFFUSION_AVATARS_MODELS_PATH="..."
DIFFUSION_AVATARS_RENDERS_PATH="..."
```

Replace the `...` with the locations where data / models / renderings should be located on your machine.

- `DIFFUSION_AVATARS_DATA_PATH`:  Location of the multi-view videos and preprocessed 3DMM meshes (See [section 2](#2-dataset) for how to obtain the dataset)
- `DIFFUSION_AVATARS_MODELS_PATH`: During training, model checkpoints and configs will be saved here
- `DIFFUSION_AVATARS_RENDERS_PATH`: Video renderings of trained models will be stored here

If you do not like creating a config file in your home directory, you can instead hard-code the paths in the [env.py](src/diffusion_avatars/env.py).

## 2. Data

Data as well as model checkpoints can be found in the [Downloads section](#4-downloads).

The folder structure assumed by the code looks as follows:
```yaml
DIFFUSION_AVATARS_DATA_PATH
├── nersemble       # Raw NeRSemble data containing RGB images, etc
│   ├── 018           # Data folder for participant 18  
│   ├── 037
│   ...
└── rendering_data  # Rasterized NPHM meshes that are the input for DiffusionAvatars 
    ├── v1.1-ID-18-nphm   # Dataset of rasterized NPHM meshes for participant 18
    ├── v1.2-ID-37-nphm
    ...
```

```yaml
DIFFUSION_AVATARS_MODELS_PATH
└── diffusion-avatars     # Folder for DiffusionAvatars checkpoints
    ├── DA-1-ID-18           # Checkpoint for DiffusionAvatars model trained on participant 18
    ├── DA-2-ID-37
    ...
```

## 3. Usage

### 3.1. Training

```shell
python scripts/train/train_diffusion_avatars.py $DATASET
```

where `$DATASET` is the identifier of a dataset folder in `${DIFFUSION_AVATARS_DATA_PATH}/rendering_data`.  
E.g., `v1.1` will train DiffusionAvatars for data of person `37` stored in the folder `v1.1-ID-37-nphm`.  
Please find the respective raw NeRSemble data as well as processed datasets in the [Downloads section](#4-downloads).

Checkpoints and train configurations will be stored in a model folder `DA-xxx` inside `${DIFFUSION_AVATARS_MODELS_PATH}/diffusion-avatars/DA-xxx-${name}`.
The incremental run id `xxx` will be automatically determined.

During training, the script will log metrics and images to *weights and biases* ([wandb.ai](http://wandb.ai)) to a project `diffusion-avatars`. 
The hold out test sequences are specified in [constants.py](src/diffusion_avatars/constants.py).

#### Memory consumption

Training takes roughly 2 days and requires at least an RTX A6000 GPU (**48GB VRAM**).
For debugging purposes, the following flags may be used to keep the GPU memory consumption below 10G:

```shell
--batch_size 1 --gradient_accumulation 4 --use_8_bit_adam --mixed_precision FP16 --dataloader_num_workers 0
```

### 3.2. Rendering

From a trained model `DA-xxx`, a self-reenactment rendering may be obtained via:

```shell
python scripts/render/render_trajectory.py DA-xxx
```

The resulting `.mp4` file is stored in `DIFFUSION_AVATARS_RENDERS_PATH`.  
Please find trained model checkpoints and corresponding raw NeRSemble data in the [Downloads section](#4-downloads).

### 3.3. Evaluation

```shell
python scripts/evaluate/evaluate.py DA-xxx
```

will evaluate the self-reenactment scenario for avatar `DA-xxx`.
Please find trained model checkpoints and corresponding processed datasets in the [Downloads section](#4-downloads).
The computed metrics and generated model predictions with paired GT images will be stored
in `${DIFFUSION_AVATARS_MODELS_PATH}/diffusion-avatars/DA-xxx-${name}/evaluations`.  
The key `"average_per_sequence_metric"` in the generated `.json` file reproduces the metrics from the paper.

### 3.4. Create custom datasets

The script 
```shell
python scripts/data/create_renderings_dataset.py $PARTICIPANT_ID $SEQUENCES
```
processes the raw NeRSemble data of `$PARTICIPANT_ID` for the comma-separated `$SEQUENCES`. 
It creates a new folder in `${DIFFUSION_AVATARS_DATA_PATH}/rendering_data` that contains the rasterized NPHM images and forms the input for training DiffusionAvatars. 
Please refer to the provided datasets in the [Downloads section](#4-downloads) for the expected folder layout for the raw NeRSemble data. 
The NPHM fittings where obtained using [MonoNPHM](https://simongiebenhain.github.io/MonoNPHM/).

### 3.5. Example Notebooks

The [notebooks folder](notebooks) contains minimal examples on how to
 - Load RGB images and NPHM renderings ([visualize_data.ipynb](notebooks/visualize_data.ipynb))
 - Load a trained model and obtain a prediction ([inference.ipynb](notebooks/inference.ipynb))

## 4. Downloads

| Participant ID | Model                                                                               | Raw NeRSemble data                                                             | Processed Data                                                                                             |
|----------------|-------------------------------------------------------------------------------------|--------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|
| 18             | [DA-1-ID-18](https://kaldir.vc.in.tum.de/diffusion_avatars/models/DA-1-ID-18.zip)   | [ID-18](https://kaldir.vc.in.tum.de/diffusion_avatars/data/nersemble/018.zip)  | [v1.1-ID-18-nphm](https://kaldir.vc.in.tum.de/diffusion_avatars/data/rendering_data/v1.1-ID-18-nphm.zip)   |
| 37             | [DA-2-ID-37](https://kaldir.vc.in.tum.de/diffusion_avatars/models/DA-2-ID-37.zip)   | [ID-37](https://kaldir.vc.in.tum.de/diffusion_avatars/data/nersemble/037.zip)  | [v1.2-ID-37-nphm](https://kaldir.vc.in.tum.de/diffusion_avatars/data/rendering_data/v1.2-ID-37-nphm.zip)   |
| 55             | [DA-3-ID-55](https://kaldir.vc.in.tum.de/diffusion_avatars/models/DA-3-ID-55.zip)   | [ID-55](https://kaldir.vc.in.tum.de/diffusion_avatars/data/nersemble/055.zip)  | [v1.3-ID-55-nphm](https://kaldir.vc.in.tum.de/diffusion_avatars/data/rendering_data/v1.3-ID-55-nphm.zip)   |
| 124            | [DA-4-ID-124](https://kaldir.vc.in.tum.de/diffusion_avatars/models/DA-4-ID-124.zip) | [ID-124](https://kaldir.vc.in.tum.de/diffusion_avatars/data/nersemble/124.zip) | [v1.4-ID-124-nphm](https://kaldir.vc.in.tum.de/diffusion_avatars/data/rendering_data/v1.4-ID-124-nphm.zip) |
| 145            | [DA-5-ID-145](https://kaldir.vc.in.tum.de/diffusion_avatars/models/DA-5-ID-145.zip) | [ID-145](https://kaldir.vc.in.tum.de/diffusion_avatars/data/nersemble/145.zip) | [v1.5-ID-145-nphm](https://kaldir.vc.in.tum.de/diffusion_avatars/data/rendering_data/v1.5-ID-145-nphm.zip) |
| 210            | [DA-6-ID-210](https://kaldir.vc.in.tum.de/diffusion_avatars/models/DA-6-ID-210.zip) | [ID-210](https://kaldir.vc.in.tum.de/diffusion_avatars/data/nersemble/210.zip) | [v1.6-ID-210-nphm](https://kaldir.vc.in.tum.de/diffusion_avatars/data/rendering_data/v1.6-ID-210-nphm.zip) |
| 251            | [DA-7-ID-251](https://kaldir.vc.in.tum.de/diffusion_avatars/models/DA-7-ID-251.zip) | [ID-251](https://kaldir.vc.in.tum.de/diffusion_avatars/data/nersemble/251.zip) | [v1.7-ID-251-nphm](https://kaldir.vc.in.tum.de/diffusion_avatars/data/rendering_data/v1.7-ID-251-nphm.zip) |
| 264            | [DA-8-ID-264](https://kaldir.vc.in.tum.de/diffusion_avatars/models/DA-8-ID-264.zip) | [ID-264](https://kaldir.vc.in.tum.de/diffusion_avatars/data/nersemble/264.zip) | [v1.8-ID-264-nphm](https://kaldir.vc.in.tum.de/diffusion_avatars/data/rendering_data/v1.8-ID-264-nphm.zip) |
 
Participant ID refers to the participants from the [NeRSemble dataset](https://tobias-kirschstein.github.io/nersemble/).
The model zip files contain a checkpoint as well as hyperparameters.
The raw NeRSemble data files contain the RGB images, segmentation masks, foreground masks, NPHM fittings (obtained with [MonoNPHM](https://simongiebenhain.github.io/MonoNPHM/)), and camera parameters.
The processed data archives contain the rasterized NPHM meshes (normals, depth, and canonical coordinates) that serve as the input for DiffusionAvatars.  
Before using the raw NeRSemble data or processed data for your own projects, please fill out the [NeRSemble dataset Terms of Use](https://docs.google.com/forms/d/e/1FAIpQLScYsXR8NVCi4nvmCbFNL0P9swsGodMnbntUJeFejtuKUMsY7Q/viewform).
<hr>

If you find our paper or code useful, please consider citing

```bibtex
@inproceedings{kirschstein2024diffusionavatars,
  title={DiffusionAvatars: Deferred Diffusion for High-fidelity 3D Head Avatars},
  author={Kirschstein, Tobias and Giebenhain, Simon and Nie{\ss}ner, Matthias},
  booktitle={Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```

Contact [Tobias Kirschstein](mailto:tobias.kirschstein@tum.de) for questions, comments and reporting bugs, or open a GitHub issue.