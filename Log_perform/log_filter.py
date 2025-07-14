import re

with open("Log_model/1_1_DDQN_ADV_TL.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

with open("Log_model/1_1_DDQN_ADV_TL.txt", "w", encoding="utf-8") as out:
    for line in lines:
        match = re.search(r"episode (\d+) is ([\d\.]+)", line)
        if match:
            episode = int(match.group(1))
            reward = float(match.group(2))
            out.write(f"Episode: {episode}, Reward: {reward}\n")