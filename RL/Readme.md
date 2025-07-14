### How to train the reinforcement learning agent
Tested with python 3.11
First install requirements.txt

Download the semantic segmentation model from [here](https://drive.google.com/file/d/1JRdPggs5jTWAXKRXk6hVxzmP-KnOr8Hw/view?usp=sharing) and place it inside Segmentation_model

Then, you can run the mario_ddqn.py file, it has multiple options. The command to train on level 1-1 using semantic segmentation, and saving weights on 1_1_ssweights is:

    python .\my_ddqn.py -ss -t -d save_model

Once it trained, if you want to see how the model behaves and plays, run:

    python .\my_ddqn.py -ss -s -d save_model -mn <your model name>

You can also change the level with --level

    python .\my_dqqn.py -ss -t -mm -d save_model -lv 1-1 5-1 7-4

For a full list of commands do:

    python .\my_ddqn.py -h