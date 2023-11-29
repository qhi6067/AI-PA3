import random
#States
states = ['R','T','D','U','8p']
#Actions 
actions = ['P', 'R', 'S','any'] #all actions are possible equally
rewards = {'P': 2,'R': 0,'S': -1,'any': 1} #States to reward
stateVal = {state: 0 for state in states} # states initialized to 0

#learning rate (alpha)0.1 per part 1; 0.2 per part 3 (Q learning)
alpha = 0.1

def monteCarlo(episodes):
    totalReward = 0
    for episode in range(episodes):
        currentState = random.choice(states)
        episodeCurrentReward = 0
        visited = set()

        while currentState != '8p':
            action = random.choice(actions) #performs random actino
            reward = rewards[action] #get dictionary value of rewards based on action
            episodeCurrentReward += reward #adding reward
            visited.add(currentState) #adding to visited set 
            currentState = random.choice(states) # chooxinga random state
            
            #Performing first visit Monte Carlo  with alpha learning rate of 0.1
            for state in visited:
                stateVal[state] = stateVal[state] + alpha * (episodeCurrentReward - stateVal[state])

            totalReward += episodeCurrentReward
        averageReward = totalReward/episodes #dividie total rewards by num of iterations. Denoted as episodes, for part 1 (50 episodes)
        print(f"Final State value: {stateVal} \n Average Reward: {averageReward}")

    #for i in states:# possible actions and rewards for each state
        #action = random.choice(actions)        # use random to get an action 
        #reward = rewards[action]        # Get the reward based on the  random selected action
        #print(f"State: {i}, Action: {action}, Reward: {reward}")
monteCarlo(50) #calling 50 episodes per part one
