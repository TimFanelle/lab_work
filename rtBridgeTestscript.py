from numpy.core.shape_base import block
import RTBridge as rtb
import matplotlib.pyplot as plt
import time
import numpy as np

pxi1 = "169.254.251.194:5555"
pxi2 = "169.254.22.65:5555"
pubPort = "5557"

bridge = rtb.BridgeSetup(pubPort, pxi2, rtb.setups.cat_12_8) 
#bridge = rtb.BridgeSetup(pubPort, pxi1, rtb.setups.hand_4_4, 20)
total_number_of_samples = 100
values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0]
slots = [0,0,0,0]
all_time_values = [0.0]*total_number_of_samples
diff_values = [0.0]*total_number_of_samples
bridge.startConnection()

try:
	count = 0
	while count<total_number_of_samples:
		'''
		for slot in range(len(slots)):
			slots[slot] = 0.7
			received = bridge.sendAndReceive(slots)
			print(str(slots)+"\r")
			slots[slot] = 0
		'''
		for item in range(len(values)):
			values[item] = 0.5
			all_time_values[count]=time.time()
			start = time.time()
			received = bridge.sendAndReceive(values)
			end = time.time()
			diff_values[count] = end-start
			print(received)
			values[item] = 0
			print(count)
		count += 1
	#plt.plot(np.array(all_time_values)/100000000.0)
	#plt.show(block=True)
	plt.plot(np.array(diff_values))
	plt.show()
except KeyboardInterrupt:
	print("\nCTRL-C Interupt detected")
finally:
	print("\nClosing the python Program")
