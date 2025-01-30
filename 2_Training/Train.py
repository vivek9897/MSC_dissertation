import numpy as np
import pandas as pd
import sys
import random
import os
import pickle
import torch
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# RL imports
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor
# import gym

# Custom imports
from Utils import *
from Environment import DirectionalChangeEnv


# load experiment parameters
with open('/Users/vivekkumar/Documents/MSC_dissertation_project/New_trainTest/params.json', 'r') as f:
    params = json.load(f)

# set seeds
seed = params['seed']
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

class ModelTrainer(object):
    """class to run training of model"""
    def __init__(self, theta):

        self.theta = theta

        # get data
        data_dict = get_data(pair, window, theta)
        train_data = data_dict['train_data']
        train_asks = data_dict['train_asks']
        train_bids = data_dict['train_bids']
        val_data = data_dict['val_data']
        val_asks = data_dict['val_asks']
        val_bids = data_dict['val_bids']

        # configure training parameters
        env_config = {
                'data': train_data, 
                'asks': train_asks, 
                'bids': train_bids
            }

        # configure algorithm
        log_path = f"/Users/vivekkumar/Documents/MSC_dissertation_project/2_Training/Logs/{self.theta}/{pair}/Window_{window}"
        env = Monitor(DirectionalChangeEnv(env_config), os.path.join(log_path, 'training_log'))
        self.model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)

    def save_checkpoint(self):
        # save model
        model_folder = f'/Users/vivekkumar/Documents/MSC_dissertation_project/New_trainTest/2_Training/Models/{self.theta}/{pair}/'
        self.model.save(model_folder + f'PPO_Window_{window}')
        print('model saved')

    def train(self):
        self.model.learn(total_timesteps=200000)
                

if __name__ == "__main__":

    # set script variables
    pair = str(sys.argv[1])
    window = int(sys.argv[2])
    theta = float(sys.argv[3])

    # create experiment
    trainer = ModelTrainer(theta)

    # train the model
    trainer.train()

    # save model
    trainer.save_checkpoint()
