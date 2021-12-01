import zmq
import zwrap
import numpy as np

class zmqWrapperBeginning:
	def __init__(self, pubPort, subIP, setup):
		self.context = zmq.Context()
		self.sub = zwrap.connectsub(self.context, "tcp://"+subIP)
		self.pub = zwrap.bindpub(self.context, "tcp://*:"+pubPort)
		self.setup = setup

	def sendAndReceive(self, activations, stepInMillisec):
		temp = []
		sending = activations
		zwrap.pubmsg(self.pub, self.setup, sending)
		#startTime = time.clock()
		#time.sleep(stepInSec)
		for _ in range(stepInMillisec):
			response = zwrap.submsg(self.sub, self.setup)
		#print("\n"+str(response))
		for value in response:
			temp.append(value)
		return temp

subIP = "169.254.251.194:5555"
pubPort = "5557"

sendOff = zmqWrapperBeginning(pubPort, subIP, zwrap.setups.hand_4_4)

values = [0.0, 0.0, 0.0, 0.0]
outputs = []

try:
	for l in np.arange(0.0, 1.02, 0.01):
		for i in np.arange(0.0, 1.02, 0.01):
			for j in np.arange(0.0, 1.02, 0.01):
				for k in np.arange(0.0, 1.01, 0.01):
					values[0] = k
					received = sendOff.sendAndReceive(values, 250)
					print("\n"+str(received))
				values[0] = 0.0
				values[1] = j
			values[0] = 0.0
			values[1] = 0.0
			values[2] = i
		values[0] = 0.0
		values[1] = 0.0
		values[2] = 0.0
		values[3] = l
	print(outputs)
except KeyboardInterrupt:
	print("\nCTRL-C Interupt detected")
finally:
	print("\nClosing the python Program")
