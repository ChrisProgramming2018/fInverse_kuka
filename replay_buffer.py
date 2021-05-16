import numpy as np
import kornia
import  sys
import torch
import torch.nn as nn
import torch.nn.functional as F
np.set_printoptions(threshold=np.inf)

class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, image_pad, device):
        self.capacity = capacity
        self.device = device

        self.aug_trans = nn.Sequential(
            nn.ReplicationPad2d(image_pad),
            kornia.augmentation.RandomCrop((obs_shape[-1], obs_shape[-1])))

        self.obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.full = False
        self.k = 0

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done, done_no_max):
        self.k +=1
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0


    def get_size():
        pass


    def sample(self, batch_size):
        idxs = np.random.randint(0, self.capacity if self.full else self.idx, size=batch_size)
    
        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]
        obses_aug = obses.copy()
        next_obses_aug = next_obses.copy()

        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        obses_aug = torch.as_tensor(obses_aug, device=self.device).float()
        next_obses_aug = torch.as_tensor(next_obses_aug,
                                         device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs],
                                           device=self.device)
        #print("state", obses)
        obses = self.aug_trans(obses)
        next_obses = self.aug_trans(next_obses)
        
        obses_aug = self.aug_trans(obses)
        next_obses_aug = self.aug_trans(next_obses)


        return obses, actions, rewards, next_obses, not_dones_no_max, obses_aug, next_obses_aug


    def get_last_k_trajectories(self):
        """    """
        if self.full:
            pass
        idxs = [x for x in range(self.idx - self.k, self.idx)]
        # print(idxs)
        obses = self.obses[idxs]
        obses_aug = obses.copy()

        obses = torch.as_tensor(obses, device=self.device).float()
        obses_aug = torch.as_tensor(obses_aug, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs],
                                           device=self.device)

        obses = self.aug_trans(obses)
        obses_aug = self.aug_trans(obses)
        self.k = 0
        return obses,  obses_aug, actions, rewards, not_dones_no_max





    def save_memory(self, filename):
        """
        Use numpy save function to store the data in a given file
        """


        with open(filename + '/obses.npy', 'wb') as f:
            np.save(f, self.obses)
        
        with open(filename + '/actions.npy', 'wb') as f:
            np.save(f, self.actions)

        with open(filename + '/next_obses.npy', 'wb') as f:
            np.save(f, self.next_obses)
        
        with open(filename + '/rewards.npy', 'wb') as f:
            np.save(f, self.rewards)
        
        with open(filename + '/not_dones.npy', 'wb') as f:
            np.save(f, self.not_dones)
        
        with open(filename + '/not_dones_no_max.npy', 'wb') as f:
            np.save(f, self.not_dones_no_max)
    
    def load_memory(self, filename):
        """
        Use numpy load function to store the data in a given file
        """


        with open(filename + '/obses.npy', 'rb') as f:
            self.obses = np.load(f)
        
        with open(filename + '/actions.npy', 'rb') as f:
            self.actions = np.load(f)

        with open(filename + '/next_obses.npy', 'rb') as f:
            self.next_obses = np.load(f)
        
        with open(filename + '/rewards.npy', 'rb') as f:
            self.rewards = np.load(f)
        
        with open(filename + '/not_dones.npy', 'rb') as f:
            self.not_dones = np.load(f)
        
        with open(filename + '/not_dones_no_max.npy', 'rb') as f:
            self.not_dones_no_max = np.load(f)
