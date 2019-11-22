from collections import deque
import numpy as np
import torch


def train(env, agent, checkpoint_name, n_episodes=2000, max_t=1000):
    """

    Params
    ======
        env (Environment): a Gym Environment
        agent: the agent to train
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = np.zeros((20,))
        for t in range(max_t):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(np.mean(score))       # save most recent score
        scores.append(score)              # save most recent score
        print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 30.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
                i_episode-100, np.mean(scores_window)))
            break

        agent.end_episode()
    return scores
