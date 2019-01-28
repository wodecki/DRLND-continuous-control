#!/usr/bin/env python
# coding: utf-8

# # Navigation - Deep Q-Network implementation



from unityagents import UnityEnvironment
import sys
import random
import torch
import numpy as np
from collections import deque
from parameters import *
import os


# Instantiate the Environment and Agent

#env = UnityEnvironment(file_name='Reacher.app')
env = UnityEnvironment(file_name='Reacher_Linux_NoVis/Reacher.x86_64')

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of actions
action_size = brain.vector_action_space_size

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]

from ddpg_agent import Agent

try:
    os.mkdir("./models")
except OSError:
    print("I can't create the 'models' directory, sorry...")

# Train the Agent with DQN
def ddpg(model_number,
        UPD, BUFFER_SIZE, BATCH_SIZE,
        LR_ACTOR, LR_CRITIC,
        fc1_units, fc2_units,
        a_gradient_clipping, a_leaky, a_dropout,
        c_gradient_clipping, c_batch_norm, c_leaky, c_dropout,
        n_episodes=400, max_t=2000, print_every=100):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        ...
    """

    agent = Agent(state_size, action_size,
                UPD,
                BUFFER_SIZE, BATCH_SIZE,
                LR_ACTOR, LR_CRITIC,
                fc1_units, fc2_units,
                a_gradient_clipping, a_leaky, a_dropout,
                c_gradient_clipping, c_batch_norm, c_leaky, c_dropout,
                0, 12345)
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=print_every)  # last 100 scores

    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        agent.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, a_dropout, a_leaky)

            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]

            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward

            if done:
                break

        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score

        with open('results.txt', 'a') as output:
            output.writelines(\
            '{}, {}, {:.2f}, {:.2f}, {}, {}, {}, {:.4f}, {:.4f}, {}, {}, {}, {}, {}, {}, {}, {}, {} \n'.format(
            model_number, i_episode, np.mean(scores_window), score,
            UPD, BUFFER_SIZE, BATCH_SIZE,
            LR_ACTOR, LR_CRITIC,
            fc1_units, fc2_units,
            a_gradient_clipping, a_leaky, a_dropout,
            c_gradient_clipping, c_batch_norm, c_leaky, c_dropout))
            output.flush()

        print('\rModel nr: {}, Episode {}, avg. score: {:.2f}, score: {:.2f}'.format\
              (model_number, i_episode, np.mean(scores_window), score), end="")
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:2f}'.format(i_episode, np.mean(scores_window), score))
        if np.mean(scores_window)>=30.0:
            with open('./models/models_solved.txt', 'a') as solved:
                solved.writelines('{}, {} \n'.format(model_number, i_episode))
                solved.flush()
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            torch.save(agent.actor_local.state_dict(), './models/checkpoint_actor_'+str(model_number)+'.pth')
            torch.save(agent.critic_local.state_dict(), './models/checkpoint_critic_'+str(model_number)+'.pth')
            break
    return scores

model_number = 1

total_no_models = len(r_UPD)*len(r_BUFFER_SIZE)*len(r_BATCH_SIZE) \
    *len(r_fc1_units)*len(r_fc2_units)*len(r_LR_ACTOR)*len(r_LR_CRITIC)\
    *len(r_a_gradient_clipping)*len(r_a_leaky)*len(r_a_dropout)\
    *len(r_LR_CRITIC)*len(r_c_gradient_clipping)*len(r_c_batch_norm)\
    *len(r_c_leaky)*len(r_c_dropout)

est_time_minutes = total_no_models*400/6
est_time_hours = est_time_minutes/60

print('Total number of models to test: ', total_no_models)
print('Estimated computing time in hours: {:.2f}\n'.format(est_time_hours))

for UPD in r_UPD:
    for BUFFER_SIZE in r_BUFFER_SIZE:
        for BATCH_SIZE in r_BATCH_SIZE:
            for fc1_units in r_fc1_units:
                for fc2_units in r_fc2_units:
                    for LR_ACTOR in r_LR_ACTOR:
                        for a_gradient_clipping in r_a_gradient_clipping:
                            for a_leaky in r_a_leaky:
                                for a_dropout in r_a_dropout:
                                    for LR_CRITIC in r_LR_CRITIC:
                                        for c_gradient_clipping in r_c_gradient_clipping:
                                            for c_batch_norm in r_c_batch_norm:
                                                for c_leaky in r_c_leaky:
                                                    for c_dropout in r_c_dropout:
                                                        scores = ddpg(model_number, int(UPD), \
                                                                int(BUFFER_SIZE), int(BATCH_SIZE), \
                                                                LR_ACTOR, LR_CRITIC, \
                                                                fc1_units, fc2_units, \
                                                                a_gradient_clipping, a_leaky, a_dropout, \
                                                                c_gradient_clipping, c_batch_norm, c_leaky, c_dropout)
                                                        model_number += 1
