# Udacity DRLND: Continuous Control

Andrzej Wodecki

January 28th, 2018



## Project details

The goal of the project is to **train the Agent to operate an arm with two joints** in the Reacher environment provided by Unity Environments. After training it should be able to stay within the target green zones for a longer time. 

**In this report I address the Option 1 of the problem (only one agent acting in the environment).**

**The state space** has 33 dimensions like the position, rotation, velocity and angular velocities of the arm.

**The action space** consists of 4 actions corresponding to torque applicable to two joints.

This is **episodic** environment. It is considered **solved** when agents gets an average score of +30 over 100 consecutive episodes.



## Getting started

First, you will need the Reacher Environment provided by Unity - the simplest way is to follow the instruction provided by Udacity and available [here](https://github.com/udacity/deep-reinforcement-learning#dependencies).

You will also need a set of python packages installed, including jupyter, numpy and pytorch. All are provided within UDACITY "drlnd" environment: follow the instructions provided eg. [here](https://github.com/udacity/deep-reinforcement-learning#dependencies). Specifically, create and activate a new environment with Python 3.6:
`conda create --name drlnd python=3.6`
`source activate drlnd`



Finally, you should have an agent simulator  be installed: for Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip), for Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip) and for Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)



## Instructions 

The structure of the code is the following:

1. *run.py* is the main code. Here all the parameters are read,  training procedures called and the results written to the appropriate files and folders.
2. *parameters.py* stores all the hyper parameters - the structure of this file is presented in more details  in the *Hyperparameter grid search* section of the *Report.md*.
3. all the results are stored in (see *Hyperparameter grid search* section of the *Report.md*):
   1. *results.txt* file
   2. *models/* subdirectory.

To run the code:

1. Specify hyperparameters in the *parameters.py*. Be careful: too many parameters may results with a very long computation time!
2. run the code by typing: *python run.py*
3. ... and check results: both on the screen and in the output files/folders.
