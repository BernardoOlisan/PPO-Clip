import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical 

class PPOBuffer:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def store(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)  

    def clear(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

class ActorNetwork(nn.Module):
    def __init__(self, input_dims, output_dim, learning_rate, 
                 fc1_dims=256, fc2_dims=256, chkpt_dir="tmp/ppo"):
        super(ActorNetwork, self).__init__()
        
        self.checkpoint_file = os.path.join(chkpt_dir, "actor_torch_ppo")
        self.actor = nn.Sequential(                
                nn.Linear(*input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, output_dim),
                nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
    def forward(self, state):
        distribution = self.actor(state)
        distribution = Categorical(distribution)

        return distribution
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        torch.load_state_dict(torch.load(self.checkpoint_file))

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, learning_rate, fc1_dims=256, 
                 fc2_dims=256, chkpt_dir="tmp/ppo"):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, "critic_torch_ppo")
        self.critic = nn.Sequential(
                nn.Linear(*input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, 1),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
    def forward(self, state):
        value = self.critic(state)

        return value 
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        torch.load_state_dict(torch.load(self.checkpoint_file))

class Agent:
    def __init__(self, n_actions, input_dims, learning_rate=3e-4, 
                 discount_factor=0.99, gae_lambda=0.95, clip=0.2, 
                 batch_size=64, n_epochs=10):
        self.discount_factor = discount_factor
        self.clip = clip 
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actor = ActorNetwork(input_dims, n_actions, learning_rate)
        self.critic = CriticNetwork(input_dims, learning_rate)
        self.buffer = PPOBuffer(batch_size)

    def remember(self, state, action, probs, vals, reward, done):
        self.buffer.store(state, action, probs, vals, reward, done)

    def save_models(self):
        print("... saving models ...")
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
    
    def load_models(self):
        print("... loading models ...")
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        state = torch.tensor(observation, dtype=torch.float)

        distribution = self.actor(state)
        value = self.critic(state)
        action = distribution.sample()

        probs = torch.squeeze(distribution.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()

        return action, probs, value 
    
    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_probs_arr, vals_arr,\
            reward_arr, dones_arr, batches = self.buffer.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount * (reward_arr[k] + self.discount_factor * values[k+1] * \
                                     (1 - int(dones_arr[k])) - values[k])
                    discount *= self.discount_factor * self.gae_lambda
                advantage[t] = a_t
            advantage = torch.tensor(advantage)

            values = torch.tensor(values)
            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float)
                old_probs = torch.tensor(old_probs_arr[batch])
                actions = torch.tensor(action_arr[batch])

                distribution = self.actor(states)
                critic_value = self.critic(states)

                new_probs = distribution.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                weighted_probs = prob_ratio * advantage[batch]
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.clip,\
                                                     1+self.clip) * advantage[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.buffer.clear()


        
        
        
        
     
            
