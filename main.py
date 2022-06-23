#imports
import numpy as np
from matplotlib import pyplot as plt
from backend import *
import pickle
from warnings import simplefilter
#import rt-bridge as rtb

#ignore warnings
simplefilter(action='ignore', category=FutureWarning)

#constants
pxi1 = "169.254.168.231"
portSend = 5555
portReceive = 5557
listeningIP = pxi1 + ":" + str(portReceive)
babbling_minutes = 2

#gathering babbling data
[babbling_kin, babbling_act] = babbling_func(babbling_min=babbling_minutes, listenAt=listeningIP, sendAt=portSend)

#train network from babbling data
network = inv_mapping_func(kinematics=babbling_kin, activations=babbling_act, early_stopping=False)

#defining cumulative variables
cum_kin = babbling_kin
cum_act = babbling_act

#save model
pickle.dump([network, cum_kin, cum_act], open("/media/tim/Chonky/Programming/VS/liveFinger/saves/liveModel.sav", 'wb'))

#Continue writing after confirming that training from babbling works