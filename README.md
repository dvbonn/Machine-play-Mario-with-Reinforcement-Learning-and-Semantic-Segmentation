# **How to use**
First, you need to download [ResNet50.pth](https://drive.google.com/file/d/1JRdPggs5jTWAXKRXk6hVxzmP-KnOr8Hw/view) and place it in the Segmentation_model folder.

After that, install requirements.txt.

Then, you can train by a simple command like
```python my_ddqn.py -ss -t -d save_model```.

To train on a specific map you can using this command<br>
```python my_ddqn.py -ss -t -d save_model -lv 5-1```.

Or multi map<br>
```python my_dqqn.py -ss -t -mm -d save_model -lv 1-1 5-1 7-4```.

After training, you can play game with the model by this command<br>
```python my_ddqn.py -ss -s -d save_model -mn <your model name>```<br>
or<br>
```python my_ddqn.py -ss -s -d save_model -mn <your model name> -lv 1-1 5-1 7-4``` for multi map.

To train with transfer learning, add the `-tl` flag :<br>
```python my_ddqn.py -ss -t -tl -d save_model -mn <pretrained_model_name>.pt -lv 1-1```.

Tested on python 3.11.

## Data Comparison Chart

To visualize and compare training results between different models or maps, use the script `Log_perform/perform_chart.py`.

### Usage

1. Place your log files (e.g., `7_4_DDQN_TL.txt`, `5_1_DQN_TL.txt`, etc.) in the `Log_perform/` directory.
2. Open a terminal in this directory and run:
   ```bash
   python perform_chart.py

# **About project**
Our project focused on how to train a RL agent using SS output as input for RL, so we do not cover training the SS model in this project. 

Our work is based on the original project from [Semantic-Segmentation-Boost-Reinforcement-Learning](https://github.com/vpulab/Semantic-Segmentation-Boost-Reinforcement-Learning/tree/69eace77a3437f98b1b437074adee5a578803581/RL)
