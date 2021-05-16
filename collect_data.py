import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import h5py
import os, sys
import argparse
import gym
from agent import TQC
from gym_grasping.envs.robot_sim_env import RobotSimEnv
from helper import FrameStack, mkdir, write_into_file
import json


def tt(ndarray, use_cuda):
  if use_cuda:
    return Variable(torch.from_numpy(ndarray).float().cuda(), requires_grad=False)
  return Variable(torch.from_numpy(ndarray).float(), requires_grad=False)





if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--env-name', default="kuka_block_grasping-v0", type=str, help='Name of a environment (set it to any Continous environment you want')
  parser.add_argument('--seed', default=True, type=bool, help='use different seed for each episode')
  parser.add_argument('--epi', default=25, type=int)
  parser.add_argument('--max_episode_steps', default=50, type=int)
  parser.add_argument('--lr-critic', default= 0.0005, type=int)               # Total number of iterations/timesteps
  parser.add_argument('--lr-actor', default= 0.0005, type=int)               # Total number of iterations/timesteps
  parser.add_argument('--lr_alpha', default=3e-4, type=float)
  parser.add_argument('--lr_decoder', default=1e-4, type=float)      # Divide by 5
  parser.add_argument('--save_model', default=True, type=bool)     # Boolean checker whether or not to save the pre-trained model
  parser.add_argument('--batch_size', default= 256, type=int)      # Size of the batch
  parser.add_argument('--discount', default=0.99, type=float)      # Discount factor gamma, used in the calculation of the total discounted reward
  parser.add_argument('--tau', default=0.005, type= float)        # Target network update rate
  parser.add_argument('--size', default=84, type=int)
  parser.add_argument('--num_q_target', default=4, type=int)    # amount of qtarget nets
  parser.add_argument('--tensorboard_freq', default=5000, type=int)    # every nth episode write in to tensorboard
  parser.add_argument('--device', default='cuda', type=str)    # amount of qtarget nets
  parser.add_argument("--n_quantiles", default=25, type=int)
  parser.add_argument("--top_quantiles_to_drop_per_net", default=2, type=int)
  parser.add_argument("--n_nets", default=5, type=int)
  parser.add_argument('--history_length', default=3, type=int)
  parser.add_argument('--image_pad', default=4, type=int)     #
  parser.add_argument('--actor_clip_gradient', default=1., type=float)     # Maximum value of the Gaussian noise added to
  parser.add_argument('--locexp', type=str)     # Maximum value
  parser.add_argument('--debug', default=False, type=bool)
  parser.add_argument('--eval', type=bool, default= True)
  parser.add_argument('--buffer_size', default=3.5e5, type=int)
  parser.add_argument('--param', default="param.json", type=str)
  args = parser.parse_args()
  
  with open (args.param, "r") as f:
      config = json.load(f)
  env= gym.make(args.env_name, renderer='egl')
  env = FrameStack(env, args)
  print(env.action_space)
  state = env.reset()
  state_dim = 200
  action_dim = 5
  config["action_dim"]= action_dim
  max_action = float(1)
  min_action = float(-1)
  config["target_entropy"]=-np.prod(action_dim)
  policy = TQC(state_dim, action_dim, max_action, config)
  directory = "pretrained/"
  if args.device == "cuda":
      use_cuda = True
  else:
      use_cuda = False
  render = False
  filename = "model-14293reward_-0.87"
  filename = directory + filename
  print("Load " , filename)
  policy.load(filename)
  policy.actor.training = False


  print("Collect data")
  replay_buffer = {"states": [], "actions": [], "next_states": [], "terminal_flags": [], "size": 0}
  scale = env.action_space.high[0]
  steps = 50
  num_ep = 20
  for i in range(num_ep):
    print("Episode %s/%s" %(i+1, num_ep))
    x = env.reset()
    state = policy.encoder.create_vector(torch.FloatTensor(x).to(args.device).div_(255).unsqueeze(0)).squeeze(0)
    for t in range(steps):
      u  = policy.select_action(np.array(x))
      xp, _, done, _ = env.step(u)
      next_state = policy.encoder.create_vector(torch.FloatTensor(xp).to(args.device).div_(255).unsqueeze(0)).squeeze(0)
      replay_buffer["states"].append(np.copy(state.cpu().detach().numpy()))
      replay_buffer["actions"].append(np.copy(u))
      replay_buffer["next_states"].append(np.copy(next_state.cpu().detach().numpy()))
      replay_buffer["terminal_flags"].append(1.0 if done else 0.0)
      replay_buffer["size"] += 1
      state = next_state
      x = xp
      if done:
        print("step", t)
        if (t+1) < steps:
           
          print("Episode ended abruptly after %s steps."%(t+1))
        break
  f = h5py.File('replay_buffer_%s_%s_%s.hdf5'%(env.unwrapped.spec.id, num_ep, np.random.randint(1000000)), 'w')
  states = np.array(replay_buffer["states"])
  actions = np.array(replay_buffer["actions"])
  next_states = np.array(replay_buffer["next_states"])
  terminal_flags = np.array(replay_buffer["terminal_flags"])
  f.create_dataset('states', states.shape, dtype=states.dtype, data=states)
  f.create_dataset('actions', actions.shape, dtype=actions.dtype, data=actions)
  f.create_dataset('next_states', next_states.shape, dtype=next_states.dtype, data=next_states)
  f.create_dataset('terminal_flags', terminal_flags.shape, dtype=terminal_flags.dtype, data=terminal_flags)
  f.close()
