import numpy as np
import gymnasium as gym
from tqdm import tqdm
from PIL import Image
import os

os.makedirs("rollouts", exist_ok=True) 

# 画像を64*64にリサイズし、正規化。
def preprocess_image(obs):
    img = Image.fromarray(obs).resize((64,64),Image.Resampling.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0

env = gym.make("CarRacing-v3", render_mode="rgb_array", continuous=True)

num_episodes = 200
max_steps = 10000

for episode in tqdm(range(num_episodes)):
    obs, info = env.reset()
    observations = []
    actions = []
    for step in range(max_steps):
        action = env.action_space.sample()
        observations.append(preprocess_image(obs))
        actions.append(action)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            break
        
    np.savez(
        f"rollouts/episode_{episode:05d}.npz",
        observations=np.array(observations, dtype=np.float32),
        actions=np.array(actions, dtype=np.float32),
    )