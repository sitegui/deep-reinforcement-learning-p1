import subprocess
import digitalocean
import dotenv
import os
import re
import pandas as pd
import concurrent.futures
import numpy as np
import json
import traceback

dotenv.load_dotenv()

manager = digitalocean.Manager(token=os.environ['DO_TOKEN'])
droplets = sorted(manager.get_all_droplets(tag_name='p3-train'), key=lambda d: d.name)


def run_cmd(droplet, cmd):
    result = subprocess.run([
        'ssh',
        '-o', 'StrictHostKeyChecking=no',
        f'root@{droplet.ip_address}',
        cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return result.stdout.decode('utf8') if result.returncode == 0 else ''


def parse_log(log):
    time = None
    episodes = None
    max_score = 0
    name = None
    for line in log.split('\n'):
        try:
            match = re.match(r'Time: (\S+?)min\s+Episode (\S+).*Avg 100: (\S+)', line)
            if match is not None:
                time = float(match.group(1))
                episodes = int(match.group(2))
                score = float(match.group(3))
                max_score = max(max_score, score)
            else:
                match = re.search(r"'name': '(.*?)'", line)
                if match is not None:
                    name = match.group(1)
        except Exception as e:
            traceback.print_exc()
    return {
        'time': time,
        'episodes': episodes,
        'max_score': max_score,
        'player': name,
    }


def ensure_model_download(droplet, player):
    local_dir = f'models/{player}'
    remote_dir = f'deep-reinforcement-learning-projects/p3/models/{player}'
    for file_name in ['params.json', 'scores.json', 'weights.pth']:
        local_file = f'{local_dir}/{file_name}'
        if not os.path.exists(local_file):
            print(f'Download {local_file}')
            os.makedirs(local_dir, exist_ok=True)
            result = subprocess.run([
                'scp',
                '-o', 'StrictHostKeyChecking=no',
                f'root@{droplet.ip_address}:{remote_dir}/{file_name}',
                local_file], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def handle_droplet(droplet):
    models_dir = 'deep-reinforcement-learning-projects/p3/models'
    finished_folders = run_cmd(droplet, f'ls {models_dir}')
    num_finished = 0
    for player in finished_folders.split('\n'):
        if player == '':
            continue
        try:
            num_finished += 1
            ensure_model_download(droplet, player)
        except Exception as e:
            traceback.print_exc()

    current_log = run_cmd(droplet, 'docker logs $(docker ps -q)')
    current_info = parse_log(current_log)
    current_info['droplet'] = droplet.name
    current_info['finished'] = num_finished

    return current_info


with concurrent.futures.ThreadPoolExecutor() as executor:
    current = pd.DataFrame(executor.map(handle_droplet, droplets)).set_index('droplet')

finished = []
for player in os.scandir('models'):
    if player.is_dir() and player.name.startswith('player-'):
        scores = json.load(open(f'{player.path}/scores.json'))
        max_score = 0
        for i in range(len(scores)-100):
            score = np.mean(scores[i:i+100])
            max_score = max(max_score, score)
        finished.append({
            'player': player.name,
            'max_score': round(max_score, 2),
            'episodes': len(scores)
        })
finished = pd.DataFrame(finished).set_index('player').sort_values('max_score', ascending=False)

print('Finished')
print(finished)

print('In progress')
print(current)
