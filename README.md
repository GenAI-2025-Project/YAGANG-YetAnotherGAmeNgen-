# YAGANG-YetAnotherGAmeNgen

We build on the work of GameNGen and show that diffusion models can learn multiple games of diffrent complexities. In this repository, we have open sourced the code for the community to build upon our work.

The models which we have trained can successfully run the following games:
- Chess
- Snake and Food
- Car Obstacle Avoid
- Conway's Game of Life (64x64 Grid)

This repository contains dataset generation files, model training and finetuning code, inference and other utilities used for documentation. We have, also, included weights of the trainined models. Inference requires 8GB of vRAM on windows computers. Inference code will run on all ARM Based MacBooks.

## How to Create the Dataset

```bash
clone
python3 chess_dataset.py
python3 snake_dataset.py
python3 car_dataset.py
python3 game_of_life_dataset.py
```
