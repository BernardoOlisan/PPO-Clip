import gym
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import copy

EPISODES = 500
EPISODE_STEPS = 200

POLICY_INPUT_DIM = 4
POLICY_HIDDEN_DIM = 32
POLICY_OUTPUT_DIM = 2
POLICY_LEARNING_RATE = 3e-4

VALUE_INPUT_DIM = 4
VALUE_HIDDEN_DIM = 32
VALUE_ACTION_DIM = 1
VALUE_LEARNING_RATE = 3e-4

EPSILON = 0.2
DISCOUNT_FACTOR = 0.99


class EpisodeBuffer:
    def __init__(self, episode_steps, state_dim, output_dim, discount_factor=0.99):
        self.states = torch.zeros(episode_steps, state_dim)
        self.actions = torch.zeros(episode_steps)
        self.policy_outputs = torch.zeros(episode_steps, output_dim)
        self.old_policy_outputs = torch.zeros(episode_steps, output_dim)
        self.discounted_rewards = torch.zeros(episode_steps)
        self.clipped_surrogate_values = torch.zeros(episode_steps)
        self.discount_factor = discount_factor
        self.episode_length = 0

    def storeExperiences(self, step, policy_output, old_policy_output, state, action, reward):
        self.states[step] = state
        self.actions[step] = action
        self.policy_outputs[step] = policy_output
        self.old_policy_outputs[step] = old_policy_output
        self.discounted_rewards[step] = reward * self.discount_factor
        self.episode_length = step

    def storeObjectivesValues(self, step, clipped_surrogate_objective):
        self.clipped_surrogate_values[step] = clipped_surrogate_objective

    def getDiscountedCumulativeRewards(self):
        return sum(self.discounted_rewards)
    
    def getExpectedSurrogateObjective(self):
        return sum(self.clipped_surrogate_values)

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forwardPropagation(self, X):
        Y = torch.relu(self.fc1(X))
        Z = self.fc2(Y)
        z_hat = F.softmax(Z, dim=-1)
        return z_hat
    
    def clippedSurrogateObjective(self, ratio, advantage):
        first_value = ratio * advantage
        second_value = torch.clamp(ratio, 1 - EPSILON, 1 + EPSILON) * advantage
        return torch.min(first_value, second_value)

    def SGA(self, expected_surrogate_objective):
        self.optimizer.zero_grad()
        loss = -expected_surrogate_objective
        loss.backward()
        self.optimizer.step()

class StateValueFunction(nn.Module):
    def __init__(self, input_dim, hidden_dim, learning_rate):
        super(StateValueFunction, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forwardPropagation(self, state):
        x = torch.relu(self.fc1(state))
        value = self.fc2(x)
        return value
    
    def MSE(self, predicted_value, target_value):
        criterion = nn.MSELoss()
        loss = criterion(predicted_value, target_value)
        return loss

    def SGD(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
class ActionValueFunction(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim, learning_rate):
        super(ActionValueFunction, self).__init__()
        self.fc1 = nn.Linear(input_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forwardPropagation(self, state, action):
        x = torch.cat((state, action.reshape(1)), dim=0)
        x = torch.relu(self.fc1(x))
        value = self.fc2(x)
        return value
    
    def MSE(self, predicted_value, target_value):
        criterion = nn.MSELoss()
        loss = criterion(predicted_value, target_value)
        return loss

    def SGD(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def main():
    policy = PolicyNetwork(POLICY_INPUT_DIM, POLICY_HIDDEN_DIM, POLICY_OUTPUT_DIM, POLICY_LEARNING_RATE) # π(s|a;0)
    old_policy = copy.deepcopy(policy) # π_old(s|a;0)
    state_value_function = StateValueFunction(VALUE_INPUT_DIM, VALUE_HIDDEN_DIM, VALUE_LEARNING_RATE) # V(s)
    action_value_function = ActionValueFunction(VALUE_INPUT_DIM, VALUE_ACTION_DIM, VALUE_HIDDEN_DIM, VALUE_LEARNING_RATE) # Q(s,a)

    expected_surrogate_objective_values = []

    env = gym.make("CartPole-v1", render_mode="human")
    for episode in range(EPISODES):
        (state, _) = env.reset()
        env.render()

        episode_buffer = EpisodeBuffer(EPISODE_STEPS, POLICY_INPUT_DIM, POLICY_OUTPUT_DIM, DISCOUNT_FACTOR)

        # Collect Experiences
        for step in range(EPISODE_STEPS):
            policy_output = policy.forwardPropagation(torch.from_numpy(state))
            old_policy_output = old_policy.forwardPropagation(torch.from_numpy(state))
            action = torch.distributions.Categorical(policy_output).sample().item()

            state, reward, terminated, truncated, info = env.step(action)

            episode_buffer.storeExperiences(step, policy_output, old_policy_output, torch.tensor(state), action, reward)

            if terminated:
                break

        # Optimization Phase
        discounted_cumulative_rewards = episode_buffer.getDiscountedCumulativeRewards()
        for episode_step in range(episode_buffer.episode_length):
            state = episode_buffer.states[episode_step]
            action = episode_buffer.actions[episode_step]
            policy_output = episode_buffer.policy_outputs[episode_step]
            old_policy_output = episode_buffer.old_policy_outputs[episode_step]

            predicted_state_value = state_value_function.forwardPropagation(state)
            predicted_action_value = action_value_function.forwardPropagation(state, action)

            loss_state_value = state_value_function.MSE(predicted_state_value, discounted_cumulative_rewards.reshape(1))
            loss_action_value = action_value_function.MSE(predicted_action_value, discounted_cumulative_rewards.reshape(1))

            state_value_function.SGD(loss_state_value)
            action_value_function.SGD(loss_action_value)

            advantage = predicted_state_value.detach() - predicted_action_value.detach()
            ratio = (policy_output + 1e-6) / (old_policy_output + 1e-6)
            clipped_surrogate_objective = policy.clippedSurrogateObjective(ratio[0], advantage)
            episode_buffer.storeObjectivesValues(episode_step, clipped_surrogate_objective)

        old_policy = copy.deepcopy(policy)

        expected_surrogate_objective = episode_buffer.getExpectedSurrogateObjective()
        expected_surrogate_objective_values.append(expected_surrogate_objective.item())
        policy.SGA(expected_surrogate_objective)

        print(f"\nEPISODE DETAILS:")
        print(f"Episode number: [{episode}]")
        print(f"Discounted Cumulative Rewards (G): {discounted_cumulative_rewards}")
        print(f"Expected Surrogate Objective (J): {expected_surrogate_objective}")

        if episode % 100 == 0:
            plt.plot(expected_surrogate_objective_values)
            plt.xlabel('Episode')
            plt.ylabel('Expected Surrogate Objective')
            plt.title('Expected Surrogate Objectives Over Time')

            plt.savefig(f'new_plots/episode_{episode}.png')
            plt.clf()

        time.sleep(1)

    env.close()

if __name__ == "__main__": 
    main()