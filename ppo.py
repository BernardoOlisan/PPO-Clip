import numpy as np 
import time
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import copy

EPISODES = 10000
EPISODE_STEPS = 100

"""
TODO:
1. If you see a good graph (losses, expected), try 10000 episodes despending
   on the results in terminated-episodes and full-episodes graphs. 
"""

NN_INPUT_DIM = 4
NN_HIDDEN_DIM = 32
NN_OUTPUT_DIM = 2
NN_LEARNING_RATE = 0.00001

VALUE_INPUT_DIM = 4
VALUE_HIDDEN_DIM = 32
VALUE_ACTION_DIM = 1
VALUE_LEARNING_RATE = 0.0001

EPSILON = 0.2


class Environment:
    def __init__(self, episodes, episode_steps):
        self.env = gym.make("CartPole-v1", render_mode="human")
        self.EPISODES = episodes
        self.EPISODE_STEPS = episode_steps

    def rollouts(self, policy, state_value_function, action_value_function):
        old_policy = copy.deepcopy(policy)

        losses = []
        expected_surrogate_objective_values = []

        for episode in range(self.EPISODES):
            print(f"###################################### EPISODE [{episode}] ######################################")

            (initial_state, _) = self.env.reset()
            self.env.render()
            state = initial_state
            clipped_surrogated_values = []

            for step in range(self.EPISODE_STEPS):
                if step == 90:
                    print("FOUND IT!")
                    break
                # print(f"---------------------- STEP [{step}] ----------------------")

                policy_output = policy.forwardPropagation(torch.from_numpy(state))
                old_policy_output = old_policy.forwardPropagation(torch.from_numpy(state))
                action = torch.distributions.Categorical(policy_output).sample().item()

                state, reward, terminated, truncated, info = self.env.step(action)

                target_value = torch.tensor([reward], dtype=torch.float32)

                predicted_state_value = state_value_function.forwardPropagation(torch.from_numpy(state))
                loss_state_value = state_value_function.MSE(predicted_state_value, target_value)
                state_value_function.SGD(loss_state_value, VALUE_LEARNING_RATE)

                predicted_action_value = action_value_function.forwardPropagation(torch.from_numpy(state), torch.tensor([action], dtype=torch.float32))
                loss_action_value = action_value_function.MSE(predicted_action_value, target_value)
                action_value_function.SGD(loss_action_value, VALUE_LEARNING_RATE)

                advantage = predicted_state_value.detach() - predicted_action_value.detach()
                ratio = (policy_output + 1e-6) / (old_policy_output + 1e-6)
                clipped_surrogate_objective = policy.clippedSurrogateObjective(ratio[0], advantage)
                clipped_surrogated_values.append(clipped_surrogate_objective)

                # # Policy Network Rollout
                # print("Policy Network (π) Rollout:")
                # print(f"Policy NN output: {policy_output}")
                # print(f"Old Policy NN output: {old_policy_output}")
                # print(f"Selected action: {action}")
                # print(f"Reward received: {reward}\n")

                # # State Value Function (V(s))
                # print("State Value Function (V(s)):")
                # print(f"Predicted Value: {predicted_state_value}")
                # print(f"Loss (V(s)): {loss_state_value}\n")

                # # Action Value Function (Q(s, a))
                # print("Action Value Function (Q(s, a)):")
                # print(f"Predicted Value: {predicted_action_value}")
                # print(f"Loss (Q(s, a)): {loss_action_value}\n")

                # # Clipped Surrogate Objective at t (J(θ_t))
                # print("Clipped Surrogate Objective at t (J(θ_t)):")
                # print(f"Ratio: {ratio[0]}")
                # print(f"Advantage: {advantage}")
                # print(f"Result: {clipped_surrogate_objective}\n")

                if terminated:
                    break

            old_policy = copy.deepcopy(policy)

            expected_surrogate_objective = sum(clipped_surrogated_values)
            expected_surrogate_objective_values.append(expected_surrogate_objective.item())
            losses.append(-expected_surrogate_objective.item())
            policy.SGA(expected_surrogate_objective, NN_LEARNING_RATE)

            print(f"\nEPISODE DETAILS:")
            print(f"Episode number: [{episode}]")
            print(f"Expected Surrogate Objective: {expected_surrogate_objective}")
            print(f"General loss: {-expected_surrogate_objective}")

            if episode % 500 == 0:
                plt.plot(expected_surrogate_objective_values, label='CLIP', marker='o', linestyle='-', color='blue')
                plt.plot(losses, label='loss', marker='x', linestyle='-', color='red')
                plt.xlabel('Episode')
                plt.ylabel('Expected Surrogate Objective')
                plt.title('Positive and Negative Surrogate Objectives Over Time')
                plt.legend()

                plt.savefig(f'plots/episode_{episode}.png')
                plt.clf()

            time.sleep(1)

        self.env.close()

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forwardPropagation(self, X):
        Y = torch.relu(self.fc1(X))
        Z = self.fc2(Y)
        z_hat = F.softmax(Z, dim=-1)
        return z_hat
    
    def clippedSurrogateObjective(self, ratio, advantage):
        first_value = ratio * advantage
        second_value = torch.clamp(ratio, 1 - EPSILON, 1 + EPSILON) * advantage
        return torch.min(first_value, second_value)

    def SGA(self, expected_surrogate_objective, learning_rate):
        optimizer = optim.SGD(self.parameters(), lr=learning_rate)
        optimizer.zero_grad()
        loss = -expected_surrogate_objective
        loss.backward()
        optimizer.step()
    

class StateValueFunction(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(StateValueFunction, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forwardPropagation(self, state):
        x = torch.relu(self.fc1(state))
        value = self.fc2(x)
        return value

    def MSE(self, predicted_value, target_value):
        criterion = nn.MSELoss()
        loss = criterion(predicted_value, target_value)
        return loss

    def SGD(self, loss, learning_rate):
        optimizer = optim.SGD(self.parameters(), lr=learning_rate)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


class ActionValueFunction(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim):
        super(ActionValueFunction, self).__init__()
        self.fc1 = nn.Linear(input_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forwardPropagation(self, state, action):
        x = torch.cat((state, action), dim=0)
        x = torch.relu(self.fc1(x))
        value = self.fc2(x)
        return value

    def MSE(self, predicted_value, target_value):
        criterion = nn.MSELoss()
        loss = criterion(predicted_value, target_value)
        return loss

    def SGD(self, loss, learning_rate):
        optimizer = optim.SGD(self.parameters(), lr=learning_rate)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    

def main():
    environment = Environment(EPISODES, EPISODE_STEPS)
    policy = PolicyNetwork(NN_INPUT_DIM, NN_HIDDEN_DIM, NN_OUTPUT_DIM) # π(s|a;0)
    state_value_function = StateValueFunction(VALUE_INPUT_DIM, VALUE_HIDDEN_DIM) # V(s)
    action_value_function = ActionValueFunction(VALUE_INPUT_DIM, VALUE_ACTION_DIM, VALUE_HIDDEN_DIM) # Q(s,a)

    environment.rollouts(policy, state_value_function, action_value_function)

if __name__ == "__main__": 
    main()
    
