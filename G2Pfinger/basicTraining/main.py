#imports
import numpy as np
from matplotlib import pyplot as plt
from fingerG2P.backend import *
import pickle
from warnings import simplefilter

#ignore warnings
simplefilter(action='ignore', category=FutureWarning)

#constants
pxi1 = "169.254.251.194"
pxi2 = "169.254.22.65"
portSend = "5557"
portReceive = 5555
listeningIP = pxi1 + ":" + str(portReceive)
babbling_minutes = 0.6

#gathering babbling data
[babbling_kin, babbling_act] = babbling_func(babbling_min=babbling_minutes, listenAt=listeningIP, sendAt=portSend, timestep=0.1)

#train network from babbling data
network = inv_mapping_func(kinematics=babbling_kin, activations=babbling_act, early_stopping=False)

#defining cumulative variables
cum_kin = babbling_kin
cum_act = babbling_act

#save model
pickle.dump([network, cum_kin, cum_act], open("/media/tim/Chonky/Programming/VS/movingFromLaptop/Mujoco_tests/results/mlp_model.sav", 'wb'))

#Continue writing after confirming that training from babbling works

np.random.seed(2)

[model, errors, cum_kin, cum_act] = initial_learning_func(model=network, babbling_kin=cum_kin, babbling_act=cum_act, num_refinements=10)
