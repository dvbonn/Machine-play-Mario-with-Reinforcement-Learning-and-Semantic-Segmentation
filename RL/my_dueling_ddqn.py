import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from tqdm import tqdm

from gym_super_mario_bros.actions import RIGHT_ONLY

import gym
import numpy as np
import collections
import cv2
from segmentator import Segmentator

import os
import argparse
import time

parser = argparse.ArgumentParser()

parser.add_argument("-s", "--show", help="Show model gameplay", action="store_true")
parser.add_argument("-lv", "--level", nargs='+',help="Level want to play/train", default=["1-1"])
parser.add_argument("-ss", "--use_segment", help="Use segmentation or not", action="store_true")
parser.add_argument("-t", "--train_mode", help="Train or not", action="store_true")
parser.add_argument("--max_exp_r", help="Maximum exploration rate", type=float, default=1.0)
parser.add_argument("--min_exp_r", help="Minimize exploration rate", type=float, default=0.02)
parser.add_argument("-e", "--episodes", help="Number of episodes", type=int, default=2500)

parser.add_argument("--relay_buffer", help="Size of experience replay buffer", type=int, default=4000)
parser.add_argument("-mm", "--multi_map", help="Train multiple maps at once", action="store_true")

parser.add_argument("-tl", "--transfer_learning", help="Transfer learning mode", action="store_true")
parser.add_argument("-p", "--play", help="Load model to play", action="store_true")
parser.add_argument("-mn", "--model_name", help="Model want to load", type=str, default="ddqn1.pt")

parser.add_argument("-d", "--dir", help="Direction to save/load model", type=str, default="Model")

args = parser.parse_args()

if args.use_segment is True:
    segmentator = Segmentator()

dir_exist = os.path.exists(args.dir) and os.path.isdir(args.dir)
if not dir_exist:
    os.mkdir(args.dir)

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        """Clear past frame buffer and init to first obs"""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs

class ProcessFrame84(gym.ObservationWrapper):
    """
    Downsamples image to 84x84
    And applies semantic segmentation if set to. Otherwise uses grayscale normal frames.
    Returns numpy array
    """
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 240 * 256 * 3:
            img_og = np.reshape(frame, [240, 256, 3]).astype(np.uint8)
            #If using semantic segmentation:
            if args.use_segment is True:
                img = segmentator.segment_labels(img_og)
                
                #Normalize labels so they are evenly distributed in values between 0 and 255 (instead of being  0,1,2,...)
                img = np.uint8(img*255/6)
            
            else:
                img = cv2.cvtColor(img_og, cv2.COLOR_BGR2GRAY) #Convert to grayscale
        else:
            assert False, "Unknown resolution."

        #Re-scale image to fit model.
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_NEAREST)
        x_t = resized_screen[18:102, :]      
        x_t = np.reshape(x_t, [84, 84, 1])

        return x_t.astype(np.uint8)

#Defines a float 32 image with a given shape and shifts color channels to be the first dimension (for pytorch)
class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]),
                                                dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)

class ScaledFloatFrame(gym.ObservationWrapper):
    """Normalize pixel values in frame --> 0 to 1"""
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0

#Stacks the latests observations along channel dimension
class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=0),
                                                old_space.high.repeat(n_steps, axis=0), dtype=dtype)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    #buffer frames. 
    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer

#Create environment (wrap it in all wrappers)
def make_env(env):
    env = MaxAndSkipEnv(env)
    #print(env.observation_space.shape)
    env = ProcessFrame84(env)
    #print(env.observation_space.shape)

    env = ImageToPyTorch(env)
    #print(env.observation_space.shape)

    env = BufferWrapper(env, 6)
    #print(env.observation_space.shape)

    env = ScaledFloatFrame(env)
    #print(env.observation_space.shape)

    return JoypadSpace(env, RIGHT_ONLY) #Fixes action sets

class DuelingDDQNModel(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self.get_conv_out(input_shape)
        # Dueling streams
        self.fc_value = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        self.fc_advantage = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def get_conv_out(self, shape):
        out = self.conv(torch.zeros(1, *shape))
        return int(np.prod(out.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        # Dueling Double DQN Q(s, A) = V(s) + A(s, a) - 1/|A| * sum(A(s, a'))
        # where V(s) is the state value and A(s, a) is the advantage
        value = self.fc_value(conv_out)
        advantage = self.fc_advantage(conv_out)
        qvals = value + advantage - advantage.mean(dim=1, keepdim=True)
        return qvals

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        probs = prios ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        batch = list(zip(*samples))
        return batch, indices, torch.tensor(weights, dtype=torch.float32)

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)

class DDQNAgent:
    def __init__(self, state_space, action_space, max_memory_size, batch_size, gamma, lr, 
                 exp_max, exp_min, exp_decay, tau=0.005):
        self.state_space = state_space
        self.action_space = action_space
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.local_net = DuelingDDQNModel(state_space, action_space).to(self.device)
        self.target_net = DuelingDDQNModel(state_space, action_space).to(self.device)
        self.target_net.load_state_dict(self.local_net.state_dict())
        self.optimizer = torch.optim.Adam(self.local_net.parameters(), lr=lr)
        self.memory = PrioritizedReplayBuffer(max_memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.exp_max = exp_max
        self.exp_min = exp_min
        self.exp_decay = exp_decay
        self.exp_rate = exp_max
        self.tau = tau
        self.step = 0
        self.ending_position = 0  # Track the ending position in the level

    def remember(self, state, action, reward, next_state, done, info=None):
        self.memory.push(state.cpu(), action.cpu(), reward.cpu(), next_state.cpu(), done.cpu())
        if info is not None:
            self.ending_position = info.get('x_pos', 0)  # Update ending position if available

    def act(self, state):
        self.step += 1
        if random.random() < self.exp_rate:
            return torch.tensor([[random.randrange(self.action_space)]])
        with torch.no_grad():
            qvals = self.local_net(state.to(self.device))
            action = torch.argmax(qvals).unsqueeze(0).unsqueeze(0).cpu()
        return action

    def soft_update(self):
        for target_param, local_param in zip(self.target_net.parameters(), self.local_net.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def experience_replay(self, beta=0.4):
        if len(self.memory) < self.batch_size:
            return
        batch, indices, weights = self.memory.sample(self.batch_size, beta)
        states = torch.cat(batch[0]).to(self.device)
        actions = torch.cat(batch[1]).long().to(self.device)
        rewards = torch.cat(batch[2]).to(self.device)
        next_states = torch.cat(batch[3]).to(self.device)
        dones = torch.cat(batch[4]).to(self.device)
        weights = weights.to(self.device).unsqueeze(1)

        q_values = self.local_net(states).gather(1, actions)
        next_actions = self.local_net(next_states).argmax(1, keepdim=True)
        next_q_values = self.target_net(next_states).gather(1, next_actions)
        expected_q = rewards + self.gamma * next_q_values * (1 - dones)

        td_errors = (q_values - expected_q.detach()).abs().detach().cpu().numpy() + 1e-6
        self.memory.update_priorities(indices, td_errors.flatten())

        loss = (F.smooth_l1_loss(q_values, expected_q.detach(), reduction='none') * weights).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.local_net.parameters(), 10)
        self.optimizer.step()
        self.soft_update()
        self.exp_rate *= self.exp_decay
        self.exp_rate = max(self.exp_rate, self.exp_min)

def show_state(env):
    cv2.imshow("Output!",env.render(mode='rgb_array')[:,:,::-1]) #Display using opencv
    cv2.waitKey(1)

def run():
    env = gym_super_mario_bros.make('SuperMarioBros-'+args.level[0]+'-v0')
    env = make_env(env)
    observation_space = env.observation_space.shape
    action_space = env.action_space.n
    agent = DDQNAgent(observation_space, action_space, args.relay_buffer, 16, 0.9, 5e-4, args.max_exp_r, args.min_exp_r, 0.99)

    env.reset()

    total_rewards = []
    ending_positions = []

    checkpoint_hours = [1, 3, 5, 10, 12]
    checkpoint_seconds = [h * 3600 for h in checkpoint_hours]
    last_checkpoint = 0
    start_time = time.time()

    for episode in tqdm(range(args.episodes)):
                #Reset state and convert to tensor
        state = env.reset()
        state = torch.Tensor(np.array([state]))

        #Set episode total reward and steps
        total_reward = 0
        steps = 0
        #Until we reach terminal state
        while True:
            #Visualize or not
            if args.show is True:
                show_state(env)
            
            #What action would the agent perform
            action = agent.act(state)
            #Increase step number
            steps += 1
            #Perform the action and advance to the next state
            state_next, reward, terminal, info = env.step(int(action[0]))
            #Update total reward
            total_reward += reward
            #Change to next state
            state_next = torch.Tensor(np.array([state_next]))
            #Change reward type to tensor (to store in ER)
            reward = torch.tensor(np.array([reward])).unsqueeze(0)

            #Is the new state a terminal state?
            terminal = torch.tensor(np.array([int(terminal)])).unsqueeze(0)

            ### Actions performed while training:
            if args.train_mode is True:
                #Add state to experience replay "dataset"
                agent.remember(state, action, reward, state_next, terminal, info)
                #Learn from experience replay.
                agent.experience_replay()

            #Update state to current one
            state = state_next

            if terminal == True:
                break #End episode loop

        #Store rewards and positions. Print total reward after episode.
        total_rewards.append(total_reward)
        ending_positions.append(agent.ending_position)
        print("Total reward after episode {} is {}".format(episode + 1, total_rewards[-1]))

        if args.train_mode is True and last_checkpoint < len(checkpoint_seconds):
            elapsed_time = time.time() - start_time
            if elapsed_time >= checkpoint_seconds[last_checkpoint]:
                checkpoint_name = f"checkpoint_{checkpoint_hours[last_checkpoint]}h.pt"
                if args.transfer_learning is True:
                    torch.save(agent.local_net.state_dict(),args.dir+ "/" + args.model_name + "_tl_ddqn_adv_1_" + checkpoint_name)
                    torch.save(agent.target_net.state_dict(),args.dir+ "/" + args.model_name + "_tl_ddqn_adv_2_" + checkpoint_name)
                else:
                    torch.save(agent.local_net.state_dict(),args.dir+ "/ddqn_adv_1_" + checkpoint_name)
                    torch.save(agent.target_net.state_dict(),args.dir+ "/ddqn_adv_2_" + checkpoint_name)
                last_checkpoint += 1

        if args.multi_map is True:
            next_level = args.level[(episode + 1) % len(args.level)]
            env = gym_super_mario_bros.make('SuperMarioBros-'+next_level+'-v0')
            env = make_env(env)
    
    if args.train_mode is True:
        if args.transfer_learning is True:
            torch.save(agent.local_net.state_dict(),args.dir+ "/" + args.model_name + "_tl_ddqn_adv_1.pt")
            torch.save(agent.target_net.state_dict(),args.dir+ "/" + args.model_name + "_tl_ddqn_adv_2.pt")
        else:
            torch.save(agent.local_net.state_dict(),args.dir+ "/ddqn_adv_1.pt")
            torch.save(agent.target_net.state_dict(),args.dir+ "/ddqn_adv_2.pt")
    env.close()

def play():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for next_level in args.level:
        env = gym_super_mario_bros.make('SuperMarioBros-'+next_level+'-v0')
        env = make_env(env)
        input_shape = env.observation_space.shape
        n_actions = env.action_space.n
        model = DuelingDDQNModel(input_shape, n_actions)
        model.load_state_dict(torch.load(args.dir + '/' + args.model_name, map_location=torch.device(device))) 
        model.eval()

        state = env.reset()
        state = torch.Tensor(np.array([state]))
        total_reward = 0
        done = False
        while not done:
            if args.show is True:
                show_state(env)
            with torch.no_grad():
                output = model(state)
                action = torch.argmax(output, dim=1).item()


            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            next_state = torch.Tensor(np.array([next_state]))

            state = next_state

        print(f"Total reward: {total_reward}")
        env.close()


if __name__ == "__main__":
    if args.play is True:
        play()
    else:
        run()