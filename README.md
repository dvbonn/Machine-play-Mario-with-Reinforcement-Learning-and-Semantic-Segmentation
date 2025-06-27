# **How to use**
First you need to download [ResNet50.pth](https://drive.google.com/file/d/1JRdPggs5jTWAXKRXk6hVxzmP-KnOr8Hw/view) and place it in the Segmentation_model folder.

After that install requirements.txt.

Then you can train by simple command like
```python my_ddqn.py -ss -t -d save_model```.

To train on specific map you can using this command
```python my_ddqn.py -ss -t -d save_model -lv 5-1```.

Or multi map
```python my_dqqn.py -ss -t -mm -d save_model -lv 1-1 5-1 7-4```.

After training, you can play game with model by this command
```python my_ddqn.py -ss -s -d save_model -mn <your model name>```
or ```python my_ddqn.py -ss -s -d save_model -mn <your model name> -lv 1-1 5-1 7-4``` for multi map.

Tested on python 3.11

# **About project**
Our project focused on how to train a RL agent using SS output as input for RL, so we do not cover training the SS model in this project. 

Our work is based on the original project from [Semantic-Segmentation-Boost-Reinforcement-Learning](https://github.com/vpulab/Semantic-Segmentation-Boost-Reinforcement-Learning/tree/69eace77a3437f98b1b437074adee5a578803581/RL)
