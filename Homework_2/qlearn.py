

import gym
import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys


if "../" not in sys.path:
  sys.path.append("../") 

from collections import defaultdict
from lib.envs.cliff_walking import CliffWalkingEnv
from lib import plotting

matplotlib.style.use('ggplot')

env = CliffWalkingEnv()

def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    
    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action. Float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA # Initialize the action space array
        best_action = np.argmax(Q[observation]) # Choose the action based on the maximum value of the policy given the observation
        A[best_action] += (1.0 - epsilon) # Update the probability of the current best action
        return A # Return the probabilities of the actions
    return policy_fn

def q_learning(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy
    
    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance to sample a random action. Float between 0 and 1.
    
    Returns:
        A tuple (Q, episode_lengths).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    Q = defaultdict(lambda: np.zeros(env.action_space.n)) # This is a dictionary that holds all the state, action values. Initially it is just full of zeros.

    # These next 3 lines are just stat trackers, not important to the actual algorithm.
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))    
    
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n) # Initialize the policy with the current Q-table, the epsilon value, and the action space. This is an epsilon greedy policy
    
    # Run for the number of episodes chosen
    for i_episode in range(num_episodes):
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()
        
        state = env.reset() # Get the current state of the environment
        
        # Run for t timesteps 
        for t in itertools.count():
            
            
            action_probs = policy(state) # Gets the probabilities of the actions from the policy. Code commented in function above.
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs) # Choose a random action, weighted by the action probabilities determined from policy
            next_state, reward, done, _ = env.step(action) # Take a step to get next state, the reward from the step, and done boolean

            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            
            
            best_next_action = np.argmax(Q[next_state]) # Using current Q table, find the next best action which is the maximum value of the state action pairs
            td_target = reward + discount_factor * Q[next_state][best_next_action] # Update the target value which is the new reward.
            td_delta = td_target - Q[state][action] # Take the difference between the target reward and the current Q state-action pair
            Q[state][action] += alpha * td_delta # Finally update the new Q table with the target difference 
                
            if done:
                break
                
            state = next_state
    
    return Q, stats # Return the final Q table and the stat info for analysis

Q, stats = q_learning(env, 500) # Run the RL environment for 500 episodes

plotting.plot_episode_stats(stats)
