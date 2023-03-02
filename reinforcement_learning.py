import gym
import numpy as np
import time

env = gym.make("FrozenLake-v1")

STATES = env.observation_space.n
ACTIONS = env.action_space.n

env.reset()

# action = env.action_space.sample()
# state = env.observation_space.sample()
# new_state, reward, done, info = env.step(action)

# env.render()

Q = np.zeros((STATES, ACTIONS))
# print(Q)

EPISODES = 2000
MAX_STEPS = 100
LEARNING_DATE = 0.81
GAMMA = 0.96

RENDER = False

epsilon = 0.9


rewards = []
for episode in range(EPISODES):
    state = env.reset()
    for _ in range(MAX_STEPS):
        if RENDER:
            env.render()
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])

        next_state, reward, done, truncated, _ = env.step(action)

        Q[state, action] = Q[state, action] + LEARNING_DATE * (reward + GAMMA * np.max(Q[next_state, :] - Q[state, action]))

        state = next_state

        if done:
            rewards.append(reward)
            epsilon -= 0.001
            break

print(Q)
print(f"Average reward: {sum(rewards)/len(rewards)}")




