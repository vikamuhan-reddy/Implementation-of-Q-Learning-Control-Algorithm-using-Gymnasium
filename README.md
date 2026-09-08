# Q-Learning Control Algorithm using Gymnasium
## Aim
To implement the **Q-Learning control algorithm** using the Gymnasium `FrozenLake-v1` environment and learn a policy for reaching the goal while avoiding holes.

## Problem Statement
The objective is to develop a model-free Q-Learning agent that learns the best action for each state through interaction with the `FrozenLake-v1` environment.
Since `is_slippery=True`, the environment is stochastic and the agent must learn a policy that performs well under uncertain movements.

## Software Requirements
- Python 3
- Jupyter Notebook / Google Colab / VS Code
- Gymnasium
- NumPy
- Matplotlib

### Installation
```bash
pip install gymnasium numpy matplotlib
```

## Environment Description
FrozenLake is a grid-world reinforcement learning environment where the agent must move from the starting point to the goal while avoiding holes.

### Grid Representation
| Symbol | Description |
|:------:|-------------|
| `S` | Starting point |
| `F` | Frozen surface |
| `H` | Hole |
| `G` | Goal |

### Environment Setup
```python
env = gym.make(
    "FrozenLake-v1",
    map_name="4x4",
    is_slippery=True
)
```
The environment contains **16 states** and **4 actions**.

### Actions
| Action | Direction |
|:------:|-----------|
| `0` | Left |
| `1` | Down |
| `2` | Right |
| `3` | Up |

### Rewards
| Event | Reward |
|-------|:------:|
| Normal movement | `0` |
| Falling into a hole | `0` |
| Reaching the goal | `1` |

## Theory
**Q-Learning** is a model-free reinforcement learning algorithm that learns the optimal action-value function without knowing the environment's transition model.
The Q-value represents the expected return for taking action `a` in state `s`.
The Q-Learning update rule is:
$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) +
\alpha
\left[
R_{t+1} +
\gamma \max_a Q(S_{t+1},a) -
Q(S_t,A_t)
\right]
$$
Where:
| Symbol | Meaning |
|--------|---------|
| $S_t$ | Current state |
| $A_t$ | Current action |
| $R_{t+1}$ | Received reward |
| $S_{t+1}$ | Next state |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |

## Epsilon-Greedy Action Selection
Epsilon-greedy balances **exploration** and **exploitation**.
With probability $\epsilon$, a random action is selected. Otherwise, the action with the highest Q-value is selected.
$$
a =
\begin{cases}
\text{Random action}, & \text{with probability } \epsilon \\
\arg\max_a Q(s,a), & \text{with probability } 1-\epsilon
\end{cases}
$$
Epsilon is gradually decreased during training so that the agent moves from exploration toward exploitation.

## Algorithm
1. Create the FrozenLake environment.
2. Initialize the Q-table with zeros.
3. Set the learning rate, discount factor, and epsilon values.
4. Reset the environment for each episode.
5. Select an action using epsilon-greedy selection.
6. Execute the action and observe the next state and reward.
7. Update the Q-value using the Q-Learning equation.
8. Continue until the episode terminates.
9. Reduce epsilon after each episode.
10. Extract the state-value function and learned policy.
11. Calculate the average reward and plot the learning curve.

## Python Program
```
# -------------------------------------------------
# Q-Learning Training
# -------------------------------------------------
episode_rewards = []
for episode in range(num_episodes):
    state, info = env.reset()
    total_reward = 0
    for step in range(100):
        action = choose_action(state, epsilon)
        next_state, reward, terminated, truncated, info = env.step(action)
        if terminated:
            target = reward
        else:
            target = reward + gamma * np.max(Q[next_state])
        Q[state, action] += learning_rate * (
            target - Q[state, action]
        )
        state = next_state
        total_reward += reward
        if terminated or truncated:
            break
    episode_rewards.append(total_reward)
    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )
# -------------------------------------------------
# State Values and Policy
# -------------------------------------------------
state_values = np.max(Q, axis=1)
learned_policy = np.argmax(Q, axis=1)

```
## Output
<img width="437" height="567" alt="image" src="https://github.com/user-attachments/assets/b4fc2ac3-ba3d-4945-9646-1e157823d5ff" />

<img width="707" height="492" alt="image" src="https://github.com/user-attachments/assets/ee8961f2-b4a8-47a7-abe7-e51939989526" />


## Result
The **Q-Learning control algorithm was successfully implemented** using the Gymnasium `FrozenLake-v1` environment. The agent learned Q-values through repeated interaction and used them to obtain a policy for navigating toward the goal.

## Inference
Q-Learning successfully learns a policy without requiring the environment's transition model. Initially, the agent explores different actions using a high epsilon value. As epsilon decreases, the agent increasingly exploits the learned Q-values.
Because `FrozenLake-v1` is stochastic when `is_slippery=True`, the agent may not succeed in every episode. However, training enables it to learn a better policy than random action selection.
