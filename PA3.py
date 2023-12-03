import random
i = 0
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
#PART 1
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
while i<100:
    print("-", end = "")
    i+=1
#need to print to Print out the values of all of the states at the end of your exp

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#Part II: Value Iteration 

def initStateValues(states):#method to initialize state values to 0 
    stateVal = {}
    for state in states:
        stateVal[state]=0
    return stateVal

def valueIteration(MDP, states):
    discountRate = 0.99  # Discount rate
    limit = 0.001    # limit
    stateValues = initStateValues(states)  # initialzing state values using the function

    episodeRewards = limit
    iterations = 0

    while episodeRewards >= limit:
        episodeRewards = 0
        for state in states:
            if state == '11am class':  # Skip terminal state
                continue

            v = stateValues[state]
            max_value = float('-inf')
            for action in MDP[state]:
                total = sum(transition['prob'] * (transition['reward'] + discountRate * stateValues[nextState])
                for nextState, transition in MDP[state][action].items()) #unpackage
                max_value = max(max_value, total)

            stateValues[state] = max_value
            episodeRewards = max(episodeRewards, abs(v - stateValues[state]))

        iterations += 1

    policy = {state: None for state in states}
    for state in states:
        if state == '11am class':  # Skip terminus
            continue

        max_action_value = float('-inf')
        best_action = None
        for action in MDP[state]:
            action_value = sum(transition['prob'] * (transition['reward'] + discountRate * stateValues[nextState])
            for nextState, transition in MDP[state][action].items())
            if action_value > max_action_value:
                max_action_value = action_value
                best_action = action

        policy[state] = best_action

    return stateValues, policy, iterations

# Print statements plus unpackage
stateValues, policy, iterations = valueIteration(MDP, states)
while i<395:
    print("-", end = "")
    i+=1
print(f"\nPart 2: Number of iterations: {iterations}")
print(f"Part 2: Final state values: {stateValues}")
print(f"Part 2: Optimal policy: {policy}")

valueIteration(MDP, states)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#Part III: Q-Learning



def q_learning(MDP, states, actions):
    alpha = 0.2
    discountRate = 0.99
    alphaDecay = 0.995
    alphaMin = 0.01
    epsilon = 0.1
    epsilonDecay = 0.995
    epsilonMin = 0.01
    # Q-values initialization
    qValues = {} # Create an empty dictionary to hold Q-values
    episode = 0

    # initialization of Q-values for each state,action 
    for state in states:
        for action in actions:
            qValues[(state, action)] = 0

    while True:
        currentState = random.choice(states)
        episodeRewards = 0  # Track the maxc change in Q values

        while currentState != '11am class': #not terminal
            # exploration vs exploitation 
            if random.random() < epsilon: # if less than epsilon (0.1) [exploration value] it chooses random action from curr state
                action = random.choice([action for action in actions if action in MDP[currentState]])
            if action == 'any': #handles error "any", rturns first option
                    action = list(MDP[currentState].keys())[0] 

            if action in MDP[currentState]: #chekc if action in curr state then unpackage to get reward, if not then choose highest Q-value
                nextState, transition = list(MDP[currentState][action].items())[0]
                reward = transition['reward']
            else:
                action = max((qValues[(currentState, a)], a) for a in actions if (currentState, a) in qValues)[1]

            #update state and reward form MDP map.
            nextState, transition = list(MDP[currentState][action].items())[0]
            reward = transition['reward']
            
            #New Q-value updated using Q learning rule adn equation: Q(s,a)←Q(s,a)+α⋅(r+γ⋅max , Q(s',a') - Q(s,a)
            prevQValue = qValues[(currentState, action)]
            maxNextQValue = max(qValues.get((nextState, a), 0) for a in actions if (nextState, a) in qValues)

            newQValue = prevQValue + alpha * (reward + discountRate * maxNextQValue - prevQValue)
            qValues[(currentState, action)] = newQValue

            # Print Q-value updates
            print(f"Episodez: {episode}\n, State: {currentState}\n Action {action}: Previous Q-Value: {prevQValue}, New Q-Value: {newQValue}\n Reward: {reward}")

            episodeRewards = max(episodeRewards, abs(newQValue - prevQValue)) # getting the highest reward
            currentState = nextState

        # Check for convergence
        if episodeRewards < 0.001:
            break

        # Decay learning and exploration rates
        alpha = max(alpha * alphaDecay, alphaMin)
        epsilon = max(epsilon * epsilonDecay, epsilonMin)
        episode +=1

    # getting policy from Q-values
    policy = {state: max((qValues.get((state, a), float('-inf')), a) for a in actions)[1] for state in states if state != '11am class'}

    return qValues, policy, episode

# Run Q-Learning
qValues, policy, totalEpisodes = q_learning(MDP, states, actions)
while i<395:
    print("-", end = "")
    i+=1

print(f"\nPart 3: Total Episodes: {totalEpisodes}")
print("Part 3: Final Q Values:", qValues)
print("Part 3: Optimal Policy:", policy)

           
