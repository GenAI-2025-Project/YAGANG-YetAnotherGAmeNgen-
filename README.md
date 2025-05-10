# YAGANG-YetAnotherGAmeNgen

[![Python 3.10](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)  
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE) 

## 🔗 Model & Dataset

You can find all of the model checkpoints and the compressed dataset in our Hugging Face repository:

[📦 YAGANG – Yet Another Game Engine on Hugging Face](https://huggingface.co/chinmayjs/YAGANG-Yet-Another-Game-Engine)

We build on the work of GameNGen and show that diffusion models can learn multiple games of diffrent complexities. In this repository, we have open sourced the code for the community to build upon our work.

The models which we have trained can successfully run the following games:
- Chess
- Snake and Food
- Car Obstacle Avoid
- Conway's Game of Life (64x64 Grid)

This repository contains dataset generation files, model training and finetuning code, inference and other utilities used for documentation. We have, also, included weights of the trainined models. Inference requires atleast 8GB of vRAM on windows computers. Inference code will run on all ARM Based MacBooks.

Generated Images:
<p align="center">
  <img src="https://github.com/user-attachments/assets/00eb359e-3cd9-4607-b0ec-e96348441156" alt="Image 1" width="30%" />
  <img src="https://github.com/user-attachments/assets/11dfb570-be7a-46b4-ac8c-e374323c58b1" alt="Image 2" width="30%" />
  <img src="https://github.com/user-attachments/assets/54efe963-45cf-44c5-9060-d9ae5c3109c4" alt="Image 3" width="30%" />
</p>


## Why a Single Diffusion Model?

- Shared Visual & Temporal Priors
   Diffusion models excel at learning pixel‑level structures and motion dynamics. A single model can capture these common patterns.

- Proven Video Diffusion Robustness
   Surveys show diffusion models achieve high fidelity and temporal consistency in video tasks. This robustness underpins stable next‑frame generation across multiple game domains.

- Foundation‑Model Simplicity
   Treating next‑frame prediction as a “foundation” task aligns with best practices in multimodal AI: one core model serves many downstream applications via lightweight conditioning. This cuts engineering overhead and streamlines deployment.

 - Efficient Auto‑Regressive Rollouts
   Research on game‑engine diffusion (e.g., DOOM) demonstrates stable, long‑duration rollouts (> 20 FPS) with a single model on modest hardware. Extending to four games leverages the same auto‑regressive denoising pipeline without per‑game retraining.

- Insights into Universal Game Dynamics
   By training on multiple games, the model reveals which visual and rule‑based features are truly shared versus domain‑specific. These insights can inform procedural generation, asset reuse, and hybrid‑genre design.


# Training

This part will document in detail the setup used to create the pipeline, dataset, and models. Follow this documentation for a quick and easy setup!

## Setup the environment

```bash
git clone https://github.com/GenAI-2025-Project/YAGANG-YetAnotherGAmeNgen-.git
cd YAGANG

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Create the Dataset

Configure the number of episodes and number of transitions inside the python dataset files. The data will be stored in ~/dataset/ folder. dataset folder will be created if it doesn't exist.

> [!WARNING]
> Current configuration of number of episodes require ~700 GB of memory -- choose the episode number depending on your local machine's capacity.

```bash
python3 chess_dataset.py
python3 snake_dataset.py
python3 car_dataset.py
python3 game_of_life_dataset.py
```
## Finetune the VAE Decoder

Stable Diffusion 1.5v VAE is trained on real life images which have curves and irregular edges. This is not ideal for our games -- the games we choose have lots of straight lines and 90 degree edges.

By Finetuning the VAE decoder, we preserve the VAE encoder's downsampling ability and enchance the decoder's upsampling ability for our game domain.

```bash
CUDA_VISIBLE_DEVICES="0" python3 finetune_decoder.py
```

## Train the Unet

We will repurpose the stable diffusion 1.5v Unet to predict next frame by sending a previous latent as input conditioned on the previous action. The code will create a new file and dump checkpoints into. Restarting training will also be done from the checkpoint. We trained the model for 10 epochs on four Nvidia H100 GPUs for 20 hours.

```bash
accelerate launch train.py
```

# Inference

1. Put the ./training_diffusion/ into your Desktop or working directory
2. Run the above inference.ipynb in the working directory.

## Usage

## Usage

| Game                     | Command              | Description                 | Controls                           |
|--------------------------|----------------------|-----------------------------|------------------------------------|
| Chess                    | `<chess>`            | Displays the starting board | UCI moves (e.g. `e2e4`, `a2a3`)    |
| Car Obstacle Avoid       | `<car>`              | Displays the starting frame | `up`, `left`, `right`              |
| Snake and Food           | `<snake_n_food>`     | Displays the starting frame | `up`, `down`, `left`, `right`      |
| Conway’s Game of Life    | `<game_of_life>`     | Displays the starting grid  | `A` (advance one step)             |
| Exit                     | `<exit>`             | Exit any game               | —                                  |


## 📖 Citation

If you use YAGANG in your research, please cite:

```bibtex
@software{yagang2025,
  author       = {Chinmay S, Rachit M,  Sharaajan G, and Anok S},
  title        = {YAGANG: Yet Another Game NGen},
  year         = 2025,
  publisher    = {GitHub},
  url          = {https://github.com/GenAI-2025-Project/YAGANG-YetAnotherGAmeNgen-}
}
```
