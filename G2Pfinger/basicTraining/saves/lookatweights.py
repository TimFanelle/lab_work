import pickle

with open("C:/Users/quest/Documents/Classes/Research/liveFinger/saves/liveModel.sav", "rb") as f:
	[model, kin, act] = pickle.load(f)

layers = model.coefs_
layer0 = layers[0]
layer1 = layers[1]
for p in layer0:
	print(p)
print('\n\n\n')
for p in layer1:
	print(p)
#print(layer0[0])
#print(layer0[1])