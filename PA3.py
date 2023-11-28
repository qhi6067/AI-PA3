import random
#States
states = ['R','T','D','U','8p']
#Actions 
actions = ['P', 'R', 'S','any']
rewards = {'P': 2,'R': 0,'S': -1,'any': 1}

def monteCarlo():
    for i in states:# possible actions and rewards for each state
        action = random.choice(actions)        # use random to get an action 
        reward = rewards[action]        # Get the reward based on the  random selected action
        print(f"State: {i}, Action: {action}, Reward: {reward}")
monteCarlo()
