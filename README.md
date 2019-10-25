# Navigation Project

This is the first project, part of Udacity's Deep Reinforcement Learning Nanodegree program, whose goal is to train a banana-catcher agent using Deep Q-Learning.

## The Environment

![the environment](./env.gif)

_(Source: https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation)

The enviroment is a large square with friendly (yellow) and dangerous (blue) bananas, the goal being to collect the largest number of yellow while avoiding the blue. To encode that, the reward at each step is defined as `+1` for picking up a yellow banana, `-1` if a blue one is collected and `0` if nothing happened.

## The Agent

The agent sensor the environment by collecting its linear forward velocity and 36 ray-based perception of objects around.

At each frame, it can decide to move forward (`0`), backward (`1`), to turn left (`2`) or right (`3`).

Each episode has a fixed duration and the final goal is to reach an average score of `+13` over 100 consecutive episodes.

## Getting Started

1. Clone this repo
2. Install `conda` and run `conda env create`.
3. Activate the newly created conda environment `deep-reinforcement-learning`
4. Follow the instructions [in the section Dependecies](https://github.com/udacity/deep-reinforcement-learning#dependencies) to install your Python environment
5. Follow the instructions [in the section Getting Started](https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation#getting-started) to install your Banana environment
6. (Optional) Install GPU support, following instructions [for Cuda](https://developer.nvidia.com/cuda-downloads) and [Pytorch](https://pytorch.org/get-started/)

## Instructions

> The README describes how to run the code in the repository, to train the agent. For additional resources on creating READMEs or using Markdown, see here and here.
