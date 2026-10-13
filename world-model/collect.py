import numpy as np
import gymnasium as gym
from tqdm import tqdm
from PIL import Image
import os
from datetime import datetime

save_dir = os.path.join("rollouts", f"{datetime.now():%Y%m%d_%H%M%S}")
os.makedirs(save_dir, exist_ok=True) 

# 画像を64*64にリサイズ。
def preprocess_image(obs):
    img = Image.fromarray(obs).resize((64,64),Image.Resampling.BILINEAR)
    return np.array(img, dtype=np.uint8)

env = gym.make("CarRacing-v3", render_mode="rgb_array", continuous=True)

num_episodes = 200
max_steps = 10000
rng = np.random.default_rng()
for episode in tqdm(range(num_episodes)):
    obs, info = env.reset()
    observations = []
    actions = []
    for step in range(max_steps):
        action_0 = rng.normal(0,0.5) # steer
        action_1 = rng.normal(0,0.2) # gas
        action_2 = rng.normal(0,0.5) # brake
        
        steer = np.tanh(action_0)
        gas = (np.tanh(action_1) + 1) / 2.0
        brake = max(np.tanh(action_2), 0.0)
        action = np.array([steer, gas, brake], dtype=np.float32)
        observations.append(preprocess_image(obs))
        actions.append(action)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            break
        
    np.savez(
        f"{save_dir}/episode_{episode:05d}.npz",
        observations=np.array(observations, dtype=np.uint8),
        actions=np.array(actions, dtype=np.float32),
    )