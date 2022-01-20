#Imports
import numpy as np
from matplotlib import pyplot as plt
from backend_functions import *
import pickle
from warnings import simplefilter

#ignore warnings
simplefilter(action='ignore', category=FutureWarning)

#define path to model
xmlpath = 'finger' #"/home/djtface/codiumFiles/Models/finger.xml"

#take user input to define in air or learning
run_mode=input("Enter 1 for air training or 2 for learning to move: ")
if int(run_mode)==1:
    babbling_simulation_minutes = 1
elif int(run_mode)==2:
    babbling_simulation_minutes = 5
else:
    raise ValueError("Invalid run mode")

#gather babbling data
[babbling_kin, babbling_act] = babbling_fcn(simulation_minutes=babbling_simulation_minutes, xmlpath=xmlpath)

#train the network using the babbling data
network = inv_mapping_func(kinematics=babbling_kin, activations=babbling_act, early_stopping=False)

#define cumulative kinematics and activations used
cum_kin = babbling_kin
cum_act = babbling_act

#save model
pickle.dump([network, cum_kin, cum_act], open("/home/djtface/codiumFiles/Mujoco_tests/results/mlp_model.sav", 'wb'))

#define seed to keep randomness the same across learning while debugging
np.random.seed(2)

#learning functions
if int(run_mode) == 1:
    [model, errors, cum_kin, cum_act] = [None, None, None, None]
elif int(run_mode) == 2:
    [best_reward, all_rewards] = learn2move_func(model=network, cum_kin=cum_kin, cum_act=cum_act, reward_thresh=6, refinement=False, Mj_render=True, xmlpath=xmlpath) #refinement initially set to False
else:
    raise ValueError("Invalid run mode")