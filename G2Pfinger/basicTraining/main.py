#imports
import numpy as np
from matplotlib import pyplot as plt
from backend import *
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
babbling_minutes = 3

run_babbling = 0
training = 0

if training == 1 and run_babbling == 1:
	#gathering babbling data
	[babbling_kin, babbling_act] = babbling_func(babbling_min=babbling_minutes, listenAt=listeningIP, sendAt=portSend, timestep=0.1)

	#train network from babbling data
	network = inv_mapping_func(kinematics=babbling_kin, activations=babbling_act, early_stopping=False)

	#defining cumulative variables
	cum_kin = babbling_kin
	cum_act = babbling_act

	#save model
	pickle.dump([network, cum_kin, cum_act], open("C:/Users/quest/Documents/Classes/Research/liveFinger/saves/checkSav.sav", 'wb'))

	#Continue writing after confirming that training from babbling works

if training == 1:
	np.random.seed(2)

	[network, errors, cum_kin, cum_act] = initial_learning_func(
		listenAt=listeningIP, sendAt=portSend, model=network, babbling_kin=cum_kin, 
		babbling_act=cum_act, num_refinements=13, timestep=0.1)

	pickle.dump([network, cum_kin, cum_act], open("C:/Users/quest/Documents/Classes/Research/liveFinger/saves/checkSav_full.sav", 'wb'))

else:
	[network, cum_kin, cum_act] = pickle.load(open("C:/Users/quest/Documents/Classes/Research/liveFinger/saves/checkSav_full.sav", 'rb'))

	_, _ = moveCycling(model=network, listenAt=listeningIP, sendAt=portSend, attempt_length=120, num_cycles=10, timestep=0.1)

print("Completed")
