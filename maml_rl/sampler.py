import gym
import torch
import multiprocessing as mp
import ipdb
from maml_rl.envs.subproc_vec_env import SubprocVecEnv
from maml_rl.episode import BatchEpisodes
from metaworld.benchmarks import ML1
import numpy as np
class MetaWorldMask(gym.Wrapper):

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.num_steps+=1
        if self.num_steps==150:
            done=  1

        if not info['success']:
            rew = 0
        info['task']=0

        return obs, rew, done, info

    def reset(self, **kwargs):
        self.num_steps=0
        return self.env.reset(**kwargs)

    def get_task(self):
        return self.env.tasks_pool[np.random.randint(0,50)]

    def sample_task(self):
        return self.env.tasks_pool[np.random.randint(0,50)]

    def reset_task(self, task):
        if task is None:
            task = self.sample_task()
        self.env.set_task(task)

            #def reset_task():
            #    pass
            #env.reset_task = reset_task
def make_env_meta(env_name):
    available_tasks = ML1.available_tasks()
    #print(available_tasks)
    if '5' in env_name:
        env = ML1.get_train_tasks(available_tasks[5])
    elif '16' in env_name:
        env = ML1.get_train_tasks(available_tasks[16])
    #print(env)
    env._max_episode_steps = 150
    tasks = env.sample_tasks(50)
    env.tasks_pool = tasks

    # def get_task():
    #     return self.tasks_pool[np.random.randint(0, 50)]
    #
    # env.get_task = get_task
    env = MetaWorldMask(env)
    env.num_steps = 0
    env._max_episode_steps = 150
    return env

def make_env(env_name):
    def _make_env():
        if 'metaworld' not in env_name:
            return gym.make(env_name)
        else:
            return make_env_meta(env_name)
    return _make_env


class BatchSampler(object):
    def __init__(self, env_name, batch_size, num_workers=mp.cpu_count() - 1):
        self.env_name = env_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.queue = mp.Queue()
        self.envs = SubprocVecEnv([make_env(env_name) for _ in range(num_workers)], queue=self.queue)
        if 'metaworld' not in env_name:
            self._env = gym.make(env_name)
        else:
            self._env = make_env_meta(env_name)
        self.total_steps = 0

    def sample(self, policy, task, params=None, gamma=0.95, device='cpu'):
        episodes = BatchEpisodes(batch_size=self.batch_size, task=task, corners=None, gamma=gamma, device=device)
        for i in range(self.batch_size):
            self.queue.put(i)
        for _ in range(self.num_workers):
            self.queue.put(None)
        observations, batch_ids = self.envs.reset()
        dones = [False]
        while (not all(dones)) or (not self.queue.empty()):
            with torch.no_grad():
                observations_tensor = torch.from_numpy(observations).type(torch.FloatTensor).to(device=device)
                action_probs_tensor = policy(observations_tensor, params['z'])
                actions_tensor = action_probs_tensor.sample()
                actions = actions_tensor.cpu().numpy()
                # TODO: Not sure need to check indexing
                action_probs = action_probs_tensor.log_prob(actions_tensor).detach().cpu().numpy()   
            new_observations, rewards, dones, new_batch_ids, infos = self.envs.step(actions)
            episodes.append(observations, actions, rewards, batch_ids, action_probs, new_observations, dones,infos)
            observations, batch_ids = new_observations, new_batch_ids
        self.total_steps += episodes.mask.sum()
        # ipdb.set_trace()
        return episodes

    def reset_task(self, task):
        tasks = [task for _ in range(self.num_workers)]
        reset = self.envs.set_task(tasks)
        return all(reset)

    def sample_tasks(self, num_tasks):
        tasks = self._env.unwrapped.sample_tasks(num_tasks)
        return tasks
