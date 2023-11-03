import numpy as np 
import time
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy

EPISODES = 5
EPISODE_STEPS = 100

NN_INPUT_DIM = 4
NN_HIDDEN_DIM = 3
NN_OUTPUT_DIM = 2
NN_LEARNING_RATE = 0.001

VALUE_INPUT_DIM = 4
VALUE_HIDDEN_DIM = 32
VALUE_ACTION_DIM = 1
VALUE_LEARNING_RATE = 0.001

EPSILON = 0.2


class Environment:
    def __init__(self, episodes, episode_steps):
        self.env = gym.make("CartPole-v1", render_mode="human")
        self.EPISODES = episodes
        self.EPISODE_STEPS = episode_steps

    def rollouts(self, policy, state_value_function, action_value_function):
        old_policy = copy.deepcopy(policy)

        for episode in range(self.EPISODES):
            print(f"###################################### EPISODE [{episode}] ######################################")

            (initial_state, _) = self.env.reset()
            self.env.render()
            state = initial_state
            clipped_surrogated_values = []

            for step in range(self.EPISODE_STEPS):
                print(f"---------------------- STEP [{step}] ----------------------")

                policy_output = policy.forwardPropagation(torch.from_numpy(state))
                old_policy_output = old_policy.forwardPropagation(torch.from_numpy(state))
                action = torch.distributions.Categorical(policy_output).sample().item()
                old_action = torch.distributions.Categorical(old_policy_output).sample().item()

                state, reward, terminated, truncated, info = self.env.step(action)

                target_value = torch.tensor([reward], dtype=torch.float32)

                predicted_state_value = state_value_function.forwardPropagation(torch.from_numpy(state))
                loss_state_value = state_value_function.MSE(predicted_state_value, target_value)
                state_value_function.SGD(loss_state_value, VALUE_LEARNING_RATE)

                predicted_action_value = action_value_function.forwardPropagation(torch.from_numpy(state), torch.tensor([action], dtype=torch.float32))
                loss_action_value = action_value_function.MSE(predicted_action_value, target_value)
                action_value_function.SGD(loss_action_value, VALUE_LEARNING_RATE)

                advantage = predicted_state_value - predicted_action_value
                ratio = torch.tensor((action + 1e-6) / (old_action + 1e-6), dtype=torch.float32)
                clipped_surrogate_objective = policy.clippedSurrogateObjective(ratio, advantage)
                clipped_surrogated_values.append(clipped_surrogate_objective)

                old_policy = copy.deepcopy(policy)

                # Policy Network Rollout
                print("Policy Network (π) Rollout:")
                print(f"Policy NN output: {policy_output}")
                print(f"Old Policy NN output: {old_policy_output}")
                print(f"Selected action: {action}")
                print(f"Reward received: {reward}\n")

                # State Value Function (V(s))
                print("State Value Function (V(s)):")
                print(f"Predicted Value: {predicted_state_value}")
                print(f"Loss (V(s)): {loss_state_value}\n")

                # Action Value Function (Q(s, a))
                print("Action Value Function (Q(s, a)):")
                print(f"Predicted Value: {predicted_action_value}")
                print(f"Loss (Q(s, a)): {loss_action_value}\n")

                # Clipped Surrogate Objective at t (J(θ_t))
                print("Clipped Surrogate Objective at t (J(θ_t)):")
                print(f"Ratio: {ratio}")
                print(f"Advantage: {advantage}")
                print(f"Result: {clipped_surrogate_objective}\n")

                if terminated:
                    time.sleep(1)
                    break
                time.sleep(0.1)

            ''' 
                The SGA computation needs to be at the end of the episode? once sum the entire episode clipped
                surrogate values steps (t)?
                I'm not even using the loss function that the papers defines, I don't know what that means...
            '''          
            expected_surrogate_objective = sum(clipped_surrogated_values)
            policy.SGA(expected_surrogate_objective, NN_LEARNING_RATE)

            print(f"\nEPISODE DETAILS:")
            print(f"Expected Surrogate Objective: {expected_surrogate_objective}")

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
        ''' Error in here...
            RuntimeError: Trying to backward through the graph a second time (or directly access saved tensors after 
            they have already been freed). Saved intermediate values of the graph are freed when you call .backward() 
            or autograd.grad(). Specify retain_graph=True if you need to backward through the graph a second time or if 
            you need to access saved tensors after calling backward.
        '''
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
    
