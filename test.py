import gym
import envs
from dqn import DQNAgent

env=gym.make('GymSumo-v0')
#env.start(gui=True)
agent = DQNAgent()
agent.train(env)




