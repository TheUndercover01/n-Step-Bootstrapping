# n-Step-Bootstrapping

Work in progress.

This repository is a submodule of another repository [TabularRL-Main](https://github.com/TheUndercover01/TabularRL-Main/tree/main), where the dynamics of the bot and environment, as well as the PyBullet simulation used, are explained in detail. 


## Overveiw

This repository contains implementations of 4 major algorithms based on the n-step bootstrapping methods method, drawing inspiration from the book "Reinforcement Learning: An Introduction" by Sutton and Barto. The algorithms implemented include n-step SARSA ,n-step Off policy Learning , n-step Tree Backup Algorithm and n-step Q(\sigma). Detailed explanations of these algorithms can be found in the accompanying Jupyter notebook file.

The repository comprises two main files: env.py and bot/robot.URDF. Within env.py, you'll find all the necessary methods required to run the environment and guide the bot through the essential steps during the simulation. On the other hand, bot/robot.URDF contains comprehensive information about the bot, encompassing its joints and links.

An image of the bot is provided below to give you an idea of its appearance.

Check out the bot on Onshape [here](https://cad.onshape.com/documents/04a8f06c4e82eef0aab52342/w/e26ea93d189b4fb4644d2868/e/ce0ae9d693e713171509edc4?renderMode=0&leftPanel=false&uiState=65b6963083efbe35d664705e).

<img src="https://github.com/TheUndercover01/TabularRL-Robotics/blob/main/image_bot.png?raw=true" alt="Robotic Arm" width="575" height="600">


## How to Run

```python
import pybullet as p
import time
import numpy as np
import pybullet_data 
import matplotlib.pyplot as plt
from Env import Pickup_Bot_Env

import math
#results 
Q = np.load('./save_nStepSARSA/3/990/Q.npy' , allow_pickle=True)
path_to_bot = './bot/robot.urdf'

policy = convert_Q_to_policy(Q.item())

Q_before_run , action_to_index = init()
    # Create environment instance
position = (0, 0.54, -0.84, -0.84) # Starting state

env = Pickup_Bot_Env(path_to_bot,position, True , False)


S_t = env.get_current_state()
    
action = env.choose_action(policy[S_t] , epsilon = 0)
   
A_t = action_to_index[action] # as we have saved a .npy the new file needs to be converted back to a dictionary
while env.rounded_position != env.terminal_state:
    env.step(action)
    reward = env.get_reward()
    
    S_t1 = env.rounded_position
    action = env.choose_action(policy[S_t1] , epsilon = 0)
  

    A_t1 = action_to_index[action]
    
    
    print("State | Action | Reward" , S_t , A_t , reward)
    print("State_next | Action_next" , S_t1 ,A_t1 )
    print("\n")
    S_t = S_t1
    A_t = A_t1


env.reset_env()
print("Reached")

```
This code can be used with any of the provided notebooks. You can adjust the starting state to any possible states mentioned in the notebook. The script simulates the bot in PyBullet and displays the bot's actions in real-time. Note that in the env.ipynb, the env.step method includes a time.sleep(1/24) line within the for loop to slow down the simulation for better visualization. You can choose to remove that line if you want to train the bot without the visualization delay.

One small thing is that the add 0.01 to while chooseing the starting state for the stand (i.e. position[1]).

## License

This project is licensed under the MIT License - see the LICENSE file for details.