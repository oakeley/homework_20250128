#!/usr/bin/env python
# coding: utf-8

# In[34]:


#!/usr/bin/env python3
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import imageio
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

GAMMA = 0.99
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
REPLAY_SIZE = 100000
TARGET_UPDATE_FREQ = 1000
MAX_EPISODES = 10000
SOLVED_REWARD = -80

class DuelingDQN(nn.Module):
    def __init__(self, input_size, n_actions):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128)
        )
        
        self.advantage = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        
        self.value = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.shared(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean(dim=1, keepdim=True)

def record_episode(env, net, device, save_path):
    """Record a single episode and save it."""
    import cv2
    obs, _ = env.reset()
    frames = []
    done = False
    
    while not done:
        state_v = torch.FloatTensor([obs]).to(device)
        with torch.no_grad():
            q_vals = net(state_v)
            action = torch.argmax(q_vals).item()
        
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        frame = env.render()
        if frame is not None:  # Ensure we have a valid frame
            # Resize to dimensions divisible by 16
            h, w = frame.shape[:2]
            new_h = ((h + 15) // 16) * 16
            new_w = ((w + 15) // 16) * 16
            frame = cv2.resize(frame, (new_w, new_h))
            frames.append(frame)
    
    if frames:  # Only save if we have frames
        imageio.mimsave(save_path, frames, fps=30)

if __name__ == "__main__":
    env = gym.make("Acrobot-v1", render_mode="rgb_array")
    
    if not os.path.exists("./training_loop"):
        os.makedirs("./training_loop")

    net = DuelingDQN(env.observation_space.shape[0], env.action_space.n).to(device)
    tgt_net = DuelingDQN(env.observation_space.shape[0], env.action_space.n).to(device)
    tgt_net.load_state_dict(net.state_dict())
    
    print(net)
    
    buffer = []
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    step_idx = 0
    done_episodes = 0

    obs, _ = env.reset()
    curr_reward = 0
    epsilon = 1.0

    while True:
        step_idx += 1
        epsilon = max(0.01, 1.0 - step_idx / 100000)

        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_v = torch.FloatTensor([obs]).to(device)
            q_vals = net(state_v)
            action = torch.argmax(q_vals).item()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        curr_reward += reward
        
        buffer.append((obs, action, reward, next_obs, done))
        if len(buffer) > REPLAY_SIZE:
            buffer.pop(0)

        if done:
            obs, _ = env.reset()
            total_rewards.append(curr_reward)
            curr_reward = 0
            done_episodes += 1
            
            mean_reward = float(np.mean(total_rewards[-100:]))
            print(f"{step_idx}: reward: {total_rewards[-1]:.2f}, mean_100: {mean_reward:.2f}, episodes: {done_episodes}, epsilon: {epsilon:.3f}")

            if done_episodes % 100 == 0:
                record_episode(
                    env,
                    net,
                    device,
                    f"./training_loop/episode_{done_episodes}.mp4"
                )

            if mean_reward > SOLVED_REWARD:
                print(f"Solved in {step_idx} steps!")
                record_episode(
                    env,
                    net,
                    device,
                    f"./training_loop/solved_episode.mp4"
                )
                break

            if done_episodes >= MAX_EPISODES:
                print(f"Stopping after {MAX_EPISODES} episodes")
                break
        else:
            obs = next_obs

        if len(buffer) < BATCH_SIZE:
            continue

        batch_indices = np.random.choice(len(buffer), BATCH_SIZE, replace=False)
        batch = [buffer[idx] for idx in batch_indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        states_v = torch.FloatTensor(states).to(device)
        actions_v = torch.LongTensor(actions).to(device)
        rewards_v = torch.FloatTensor(rewards).to(device)
        next_states_v = torch.FloatTensor(next_states).to(device)
        done_mask = torch.BoolTensor(dones).to(device)

        state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            next_state_values = tgt_net(next_states_v).max(1)[0]
            next_state_values[done_mask] = 0.0
            expected_state_action_values = rewards_v + GAMMA * next_state_values

        optimizer.zero_grad()
        loss_v = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        loss_v.backward()
        optimizer.step()

        if step_idx % TARGET_UPDATE_FREQ == 0:
            tgt_net.load_state_dict(net.state_dict())

    env.close()


# In[ ]:





# In[ ]:




