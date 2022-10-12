import gym
from stable_baselines3 import A2C
from stable_baselines3 import PPO
import os

models_dir = "models/A2C"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)


env = gym.make("LunarLander-v2")

env.reset()

model = A2C(policy='MlpPolicy', verbose=1, env=env, tensorboard_log=logdir)
TIMESTAMPS = 10000

for i in range(1,30):
    model.learn(total_timesteps=TIMESTAMPS, reset_num_timesteps=False, tb_log_name="A2C")
    model.save(f"{models_dir}/{TIMESTAMPS*i}")
"""episodes = 10

for _ in range(episodes):
    observation = env.reset()
    done = False
    while not done:
        env.render()
        observation, reward, done, *info = env.step(env.action_space.sample())"""

env.close()