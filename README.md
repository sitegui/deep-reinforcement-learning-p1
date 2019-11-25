# Continuous Control

This is the second project, part of Udacity's Deep Reinforcement Learning Nanodegree program, whose goal is to train an arm to move and follow a moving target area.

## The Environment

![the environment](./env.gif)

_(Source: https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control)_

The environment is made of 20 independent agent arms, each of one them with 4 continous joints. A reward of `+0.1` is provided at each step when the agent's hand is inside the delimited area. The goal is to keep it that way for as long as possible.

## The Agent

The agent sensors the environment by collecting 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm.

At each frame, a torque is applied to its four joints. Each value should be between -1 and 1.

Each episode has a fixed duration and the final goal is to reach an average score of `+30` over 100 consecutive episodes.

## Getting Started

1. Clone this repo
2. Install `conda` and run `conda env create`.
3. Activate the newly created conda environment `deep-reinforcement-learning`
4. Follow the instructions [in the section Dependecies](https://github.com/udacity/deep-reinforcement-learning#dependencies) to install your Python environment
5. Follow the instructions [in the section Getting Started](https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control#getting-started) to install your Reacher environment (version 2)
6. (Optional) Install GPU support, following instructions [for Cuda](https://developer.nvidia.com/cuda-downloads) and [Pytorch](https://pytorch.org/get-started/locally/)

## Instructions

To explore locally, you'll need to start your Jupyter server with `jupyter notebook`.

The training code and related experiments are placed in the [Navigation](./Navigation.ipynb) notebook.
If you just want to see the final agent, the [Result](./Result.ipynb) is the way to go.
