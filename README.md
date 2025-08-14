# Machine play Mario with Reinforcement-Learning and Semantic Segmentation

[![Generic badge](https://img.shields.io/badge/Made_with-Python-blue.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/Made_with-Kaggle-orange.svg)](https://shields.io/)
[![PyPI license](https://img.shields.io/pypi/l/ansicolortags.svg)](https://pypi.python.org/pypi/ansicolortags/)

<p align="center">
  <img src="./Images/1-1-mario.gif" width="300"/>
  <img src="./Images/5-1-mario.gif" width="300"/>
</p>
<p align="center">
    <em>Mario after training.</em>
</p>

## **About project**
Our project focused on how to train a RL agent using SS output as input for RL, so we do not cover training the SS model in this project. 

You can see our full project and experiments on Kaggle:  
[Mario Project on Kaggle](https://www.kaggle.com/code/kamuisi/mario-project)

Our work is based on the original project from [Semantic-Segmentation-Boost-Reinforcement-Learning](https://github.com/vpulab/Semantic-Segmentation-Boost-Reinforcement-Learning/tree/69eace77a3437f98b1b437074adee5a578803581/RL)

---

## **Research Approach**

In this project, we systematically explored and improved deep reinforcement learning algorithms for the Super Mario Bros environment. Our research began with the implementation of the Deep Q-Network (DQN) algorithm as a baseline. We then enhanced the agent's performance and stability by adopting the Double DQN (DDQN) approach, which addresses the overestimation bias of Q-values present in standard DQN. Finally, we further advanced our model by applying the Dueling DDQN architecture, which separates the estimation of state value and advantage, leading to more robust learning and improved results.

This step-by-step progression allowed us to analyze the strengths and weaknesses of each algorithm and demonstrate the benefits of each improvement in the context of training an agent to play Mario.

---

## **Key Algorithms and Their Equations**

### 1. Deep Q-Network (DQN)

The DQN algorithm updates the Q-value using the Bellman equation:

\[
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right]
\]

In practice, the loss function minimized is:

\[
L(\theta) = \mathbb{E}_{(s,a,r,s')} \left[ \left( r + \gamma \max_{a'} Q_{\theta^-}(s', a') - Q_\theta(s, a) \right)^2 \right]
\]

where \( \theta \) are the parameters of the online network and \( \theta^- \) are the parameters of the target network.

---

### 2. Double DQN (DDQN)

Double DQN reduces overestimation by decoupling action selection and evaluation:

\[
y^{DDQN} = r + \gamma Q_{\theta^-}\left(s', \arg\max_{a'} Q_\theta(s', a')\right)
\]

The loss function is:

\[
L(\theta) = \mathbb{E}_{(s,a,r,s')} \left[ \left( y^{DDQN} - Q_\theta(s, a) \right)^2 \right]
\]

---

### 3. Dueling DDQN

Dueling DDQN separates the estimation of the state value \( V(s) \) and the advantage \( A(s, a) \):

\[
Q(s, a; \theta, \alpha, \beta) = V(s; \theta, \beta) + \left( A(s, a; \theta, \alpha) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s, a'; \theta, \alpha) \right)
\]

where \( \theta \) are the shared parameters, \( \alpha \) and \( \beta \) are parameters of the advantage and value streams, respectively.

---

These mathematical formulations are the foundation for the improvements we implemented and tested in our Mario RL