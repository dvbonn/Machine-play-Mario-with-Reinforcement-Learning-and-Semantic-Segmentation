import re

with open("5_1_DQN_TL.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

with open("5_1_DQN_TL.txt", "w", encoding="utf-8") as out:
    for line in lines:
        match = re.search(r"episode (\d+) is ([\d\.]+)", line)
        if match:
            episode = int(match.group(1))
            reward = float(match.group(2))
            out.write(f"Episode: {episode}, Reward: {reward}\n")