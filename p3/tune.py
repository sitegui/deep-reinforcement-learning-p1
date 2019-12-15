import random
import json
import os
import digitalocean
import dotenv

dotenv.load_dotenv()

region = 'fra1'
size = 'c-4'
image = '56350189'
token = os.environ['DO_TOKEN']
ssh_keys = 'd3:5f:69:0e:50:15:2e:b3:49:fd:92:d0:25:a8:c6:36,cc:5e:c0:80:13:c1:f5:7d:10:d6:db:5c:9b:05:06:7d'
commit = 'e07ae348490258fc0036c1ea10283a2c5cfbfce4'

params_spec = {
    'max_minutes': [120],
    'max_episodes': [3000],
    'memory_size': [int(1e4), int(5e4), int(1e5), int(5e5), int(1e6)],
    'warm_up': [int(1e3), int(5e3), int(1e4), int(5e4), int(1e5)],
    'batch_size': [256, 512, 1024, 2048, 4096],
    'discount': [0.95, 0.99, 0.995, 0.999, 0.9995],
    'tau': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
    'gradient_clip': [0.25, 0.5, 1, 2, 4],
    'random_process': ['gaussian', 'ou'],
    'random_theta': [0.025, 0.05, 0.1, 0.2, 0.4],
    'random_std': [0.5, 0.75, 1, 1.25, 1.5],
    'random_std_decay': [0.99, 0.995, 0.999, 0.9995, 0.9999],
    'update_every': [1, 2, 4, 8, 16],
    'update_epochs': [1, 2, 4, 8, 16],
    'h1_size': [32, 64, 128, 256, 512],
    'h2_size': [32, 64, 128, 256, 512],
    'actor_lr': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
    'critic_lr': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
}

num_runs = 50
runs_per_node = 5
runs = []

while len(runs) < num_runs:
    # Chose each param at random
    params = {
        key: random.choice(values)
        for key, values in params_spec.items()
    }
    params['name'] = f'player-{len(runs):02d}'

    # Impose some constraints
    if params['warm_up'] > params['memory_size']:
        continue
    if params['random_process'] == 'gaussian' and params['random_theta'] != 0.1:
        continue

    args = ' '.join(f'--{key} {value}' for key, value in params.items())
    runs.append(f'docker run -v "$PWD/models:/app/models" p3 {args}')

for i in range(0, num_runs, runs_per_node):
    node_runs = runs[i:i+runs_per_node]
    droplet_commands = '\n'.join(node_runs)

    droplet_name = f'p3-train-{i/runs_per_node:02.0f}'
    user_data = f'''#!/bin/bash
    cd /root/deep-reinforcement-learning-projects/p3
    git fetch
    git checkout --force {commit}
    docker build -t p3 .
    {droplet_commands}
    '''
    droplet = digitalocean.Droplet(
        token=token,
        name=droplet_name,
        region=region,
        size=size,
        image=image,
        ssh_keys=ssh_keys.split(','),
        user_data=user_data,
        tags=['p3-train']
    )
    droplet.create()
    print(f'Created droplet {droplet.id}')
