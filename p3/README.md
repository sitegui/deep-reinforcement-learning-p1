# Multi-Agent Tennis

This is the third project, part of Udacity's Deep Reinforcement Learning Nanodegree program, whose goal is to train an agent to play tennis, keeping the ball from falling into the ground.

## The Environment

![the environment](./env.gif)

_(Source: https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet)_

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

## Getting Started

1. Clone this repo
2. Install `conda` and run `conda env create`.
3. Activate the newly created conda environment `deep-reinforcement-learning`
4. Follow the instructions [in the section Dependecies](https://github.com/udacity/deep-reinforcement-learning#dependencies) to install your Python environment
5. Follow the instructions [in the section Getting Started](https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control#getting-started) to install your Reacher environment (version 2)
6. (Optional) Install GPU support, following instructions [for Cuda](https://developer.nvidia.com/cuda-downloads) and [Pytorch](https://pytorch.org/get-started/locally/)

## Instructions

To explore locally, you'll need to start your Jupyter server with `jupyter notebook`.

The training code and related experiments are placed in the [Tennis](./Tennis.ipynb) notebook.
If you just want to see the final agent, the [Result](./Result.ipynb) is the way to go.
