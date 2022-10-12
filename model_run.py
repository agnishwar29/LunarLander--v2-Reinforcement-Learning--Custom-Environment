import gym
from stable_baselines3 import PPO, A2C

env = gym.make("LunarLander-v2")
env.reset()

models_dir = "models/PPO"
model_path = f"{models_dir}/200000.zip"
model = PPO.load(model_path, env=env)

episodes = 10

for ep in range(episodes):
    observation = env.reset()
    done = False
    while not done:
        env.render()
        action, _ = model.predict(observation)
        observation, reward, done, info = env.step(action)

