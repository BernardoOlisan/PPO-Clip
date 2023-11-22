import gym 
import numpy as np
from ppo import Agent
import matplotlib.pyplot as plt

EPISODES = 300
N = 20
BATCH_SIZE = 5
N_EPOCHS = 4

LEARNING_RATE = 3e-4
DISCOUNT_FACTOR = 0.99
CLIP = 0.2
GAE_LAMBDA = 0.95

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    agent = Agent(env.action_space.n, env.observation_space.shape, LEARNING_RATE,
                  DISCOUNT_FACTOR, GAE_LAMBDA, CLIP, BATCH_SIZE, N_EPOCHS)

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for episode in range(EPISODES):
        (observation, _) = env.reset()
        done = False 
        truncated = False
        score = 0
        while not done or not truncated:
            action, prob, value = agent.choose_action(observation)
            observation_, reward, done, truncated, info = env.step(action)
            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, value, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode', episode, 'score %.1f' % score, 'avg score %.1f' % avg_score,
              'time_steps', n_steps, 'learning_steps', learn_iters)
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, "plots/cumulative_score.png")
