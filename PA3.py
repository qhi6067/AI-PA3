#update 1.5
import random
#States
states = ['RU 8p', 'TU 10p', 'RU 10p', 'RD 10p', 'RU 8a', 'RD 8a', 'TU 10a', 'RU 10a', 'RD 10a', 'TD 10a', '11am class']
#Actions 
actions = ['P', 'R', 'S','any'] #all actions are possible equally
rewards = {'P': 2,'R': 0,'S': -1,'any': 1} #States to reward
#Transition Probability
MDP = {
    'RU 8p': {
        'P': {'TU 10p': {'prob': 1.0, 'reward': 2.0}},
        'R': {'RU 10p': {'prob': 1.0, 'reward': 0.0}},
        'S': {'RD 10p': {'prob': 1.0, 'reward': -1.0}}
    },
    'TU 10p': {
        'P': {'RU 10a': {'prob': 1.0, 'reward': 2.0}},
        'R': {'RU 8a': {'prob': 1.0, 'reward': 0.0}}
    },
    'RU 10p': {
        'R': {'RU 8a': {'prob': 1.0, 'reward': 0.0}},
        'P': {'RU 8a': {'prob': 0.5, 'reward': 2.0}, 'RU 10a': {'prob': 0.5, 'reward': 2.0}},
        'S': {'RD 8a': {'prob': 1.0, 'reward': 0.0}}
    },
    'RD 10p': {
        'R': {'RU 8a': {'prob': 1.0, 'reward': 0.0}},
        'P': {'RD 8a': {'prob': 0.5, 'reward': 2.0}, 'RD 10a': {'prob': 0.5, 'reward': 2.0}}
    },
    'RU 8a': {
        'P': {'TU 10a': {'prob': 1.0, 'reward': 2.0}},
        'R': {'RU 10a': {'prob': 1.0, 'reward': 0.0}},
        'S': {'RD 10a': {'prob': 1.0, 'reward': -1.0}}   
    },
    'RD 8a': {
        'R': {'RD 10a': {'prob': 1.0, 'reward': 0.0}},
        'P': {'TD 10a': {'prob': 1.0, 'reward': 2.0}}
    },
    'TU 10a': {
        'any': {'11am class': {'prob': 1.0, 'reward': -1.0}}
    },
    'RU 10a': {
        'any': {'11am class': {'prob': 1.0, 'reward': 0.0}}
    },
    'RD 10a': {
        'any': {'11am class': {'prob': 1.0, 'reward': 4.0}}
    },
    'TD 10a': {
        'any': {'11am class': {'prob': 1.0, 'reward': 3.0}}
    }
}


stateVal = {state: 0 for state in states} # states initialized to 0

#learning rate (alpha)0.1 per part 1; 0.2 per part 3 (Q learning)
alpha = 0.1

#modification to choose next state based on transition probabilities
def monteCarlo(episodes):
    for episode in range(episodes):
        currentState = random.choice(states)
        episodeRewards = 0
        visited = set()
        stateActionPairs = []

        while currentState != '11am class':
            # Debugging print 
            #print(f"Current State: {currentState}")

            available_actions = [action for action in actions if action in MDP[currentState]]
            if not available_actions:
                #Debugging print
                #print(f"No available actions for state: {currentState}")
                break  # Exit loop if no actions available

            action = random.choice(available_actions)
            nextState, transition = list(MDP[currentState][action].items())[0]
            reward = transition['reward']
            episodeRewards += reward
            stateActionPairs.append((currentState, action, reward))

            currentState = nextState

        # First-visit Monte Carlo updates
        for state, action, reward in stateActionPairs:
            if state not in visited:
                stateVal[state] += alpha * (episodeRewards - stateVal[state])
                visited.add(state)

        # Print the episode's sequence of experience and rewards
        print(f"Episode {episode}: {stateActionPairs}, Total Reward: {episodeRewards}")

    # Calculate and print average reward
    averageReward = sum(episodeRewards for _ in range(episodes)) / episodes
    print(f"Final State Values: {stateVal}, Average Reward: {averageReward}")

monteCarlo(50)
#need to print to Print out the values of all of the states at the end of your exp