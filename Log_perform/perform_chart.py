import matplotlib.pyplot as plt
import numpy as np
import os

def load_data(filename):
    episodes = []
    rewards = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 2:
                ep = int(parts[0].split(':')[1].strip())
                rw = float(parts[1].split(':')[1].strip())
                episodes.append(ep)
                rewards.append(rw)
    sorted_pairs = sorted(zip(episodes, rewards), key=lambda x: x[0])
    episodes_sorted, rewards_sorted = zip(*sorted_pairs)
    return np.array(episodes_sorted), np.array(rewards_sorted)

# Dữ liệu map chính
filename1 = "7_4_DDQN_TL.txt"
map_name1 = os.path.splitext(os.path.basename(filename1))[0]
episodes1, rewards1 = load_data(filename1)
window = 200
rewards_ma1 = np.convolve(rewards1, np.ones(window)/window, mode='valid')

# Dữ liệu map so sánh
filename2 = "5_1_DQN_TL.txt"
map_name2 = os.path.splitext(os.path.basename(filename2))[0]
episodes2, rewards2 = load_data(filename2)
rewards_ma2 = np.convolve(rewards2, np.ones(window)/window, mode='valid')

filename3 = "5_1_DDQN_TL.txt"
map_name3 = os.path.splitext(os.path.basename(filename3))[0]
episodes3, rewards3 = load_data(filename3)
rewards_ma3 = np.convolve(rewards3, np.ones(window)/window, mode='valid')

filename4 = "7_4_DQN_TL.txt"
map_name4 = os.path.splitext(os.path.basename(filename4))[0]
episodes4, rewards4 = load_data(filename4)
rewards_ma4 = np.convolve(rewards4, np.ones(window)/window, mode='valid')

plt.style.use('seaborn-v0_8-darkgrid')
plt.figure(figsize=(14, 7))

plt.plot(episodes1[window-1:], rewards_ma1, label=f'{map_name1}', color='crimson', linewidth=2.5)
plt.plot(episodes2[window-1:], rewards_ma2, label=f'{map_name2}', color='royalblue', linewidth=2.5)
plt.plot(episodes3[window-1:], rewards_ma3, label=f'{map_name3}', color='gold', linewidth=2.5)
plt.plot(episodes4[window-1:], rewards_ma4, label=f'{map_name4}', color='green', linewidth=2.5)

mean_reward1 = np.mean(rewards1)
mean_reward2 = np.mean(rewards2)
mean_reward3 = np.mean(rewards3)
mean_reward4 = np.mean(rewards4)
plt.axhline(mean_reward1, color='crimson', linestyle='--', linewidth=1.2, label=f'{map_name1} Average ({mean_reward1:.1f})')
plt.axhline(mean_reward2, color='royalblue', linestyle='--', linewidth=1.2, label=f'{map_name2} Average ({mean_reward2:.1f})')
plt.axhline(mean_reward3, color='gold', linestyle='--', linewidth=1.2, label=f'{map_name3} Average ({mean_reward3:.1f})')
plt.axhline(mean_reward4, color='green', linestyle='--', linewidth=1.2, label=f'{map_name4} Average ({mean_reward4:.1f})')

plt.xlabel('Episode', fontsize=14)
plt.ylabel('Reward', fontsize=14)
plt.title(f'Model RawPixel after Transfer Learning ', fontsize=18, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()