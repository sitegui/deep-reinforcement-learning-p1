import ddpg_agent
from unityagents import UnityEnvironment
import numpy as np
import argparse
import time
env = UnityEnvironment(file_name="Tennis_Linux_NoVis/Tennis.x86")


brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]
action_size = brain.vector_action_space_size
states = env_info.vector_observations
state_size = states.shape[1]


class GymEnvWrapper:
    """ Wrap an Unity Environment to provide the familiar OpenGym interface """

    def __init__(self, unity_env, brain_name):
        self.unity_env = unity_env
        self.brain_name = brain_name

    def reset(self):
        env_info = self.unity_env.reset(train_mode=True)[self.brain_name]
        return env_info.vector_observations

    def step(self, action):
        env_info = self.unity_env.step(action)[self.brain_name]
        next_state = env_info.vector_observations
        reward = env_info.rewards
        done = env_info.local_done
        return next_state, reward, done, {}


gym_env = GymEnvWrapper(env, brain_name)


def train(agent, max_minutes, max_episodes):
    start = time.time()
    while True:
        agent.train_step()
        if time.time() - start > max_minutes * 60 or agent.episodes > max_episodes:
            break
        if agent.episodes >= 100:
            avg_score = np.mean(agent.scores_window)
            if avg_score >= 0.5:
                print(
                    f'Environment solved in {agent.episodes - 100} episodes!\tAverage Score: {avg_score:.2f}')
                break
    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_minutes', type=int, default=60)
    parser.add_argument('--max_episodes', type=int, default=2000)
    parser.add_argument('--name', type=str, default='player')
    parser.add_argument('--memory_size', type=int, default=int(1e5))
    parser.add_argument('--warm_up', type=int, default=int(1e4))
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--discount', type=float, default=0.995)
    parser.add_argument('--tau', type=float, default=1e-3)
    parser.add_argument('--gradient_clip', type=float, default=1)
    parser.add_argument('--random_process', type=str, default='gaussian')
    parser.add_argument('--random_theta', type=float, default=0.1)
    parser.add_argument('--random_std', type=float, default=1)
    parser.add_argument('--random_std_decay', type=float, default=0.999)
    parser.add_argument('--update_every', type=int, default=5)
    parser.add_argument('--update_epochs', type=int, default=5)
    parser.add_argument('--h1_size', type=int, default=256)
    parser.add_argument('--h2_size', type=int, default=256)
    parser.add_argument('--actor_lr', type=float, default=1e-3)
    parser.add_argument('--critic_lr', type=float, default=1e-4)
    args = parser.parse_args()
    params = args.__dict__

    print(f'Train agent with params: {params}')
    agent = train(ddpg_agent.Agent(gym_env, state_size, action_size, params), args.max_minutes, args.max_episodes)
    agent.save()
