import numpy as np
#from sklearn.neural_network import MLPRegressor
import tensorflow as tf
from tensorflow import keras
#import tensorflow
from tensorflow.keras import layers
from keras.models import load_model

excursions = []

with open('/media/tim/Chonky/Programming/VS/movingFromLaptop/softModel/excursions.csv') as ex:
    for t in ex:
        temp = t.split(';')
        for i in range(len(temp)):
            temp[i] = float(temp[i])
        excursions.append(temp)

excursions = np.array(excursions)
#print(excursions)

endpoints = []
with open('/media/tim/Chonky/Programming/VS/movingFromLaptop/softModel/endpoints.csv') as ep:
    for t in ep:
        temp = t.split(';')
        for i in range(len(temp)):
            temp[i] = float(temp[i])
        endpoints.append(temp)

endpoints = np.array(endpoints)
#print(endpoints)

'''
model = MLPRegressor(
    hidden_layer_sizes=17, activation='identity', verbose=True, solver='lbfgs',
    warm_start=True, early_stopping=False, learning_rate='adaptive', max_iter=1000
)
model.fit(endpoints, excursions)
est_exc = model.predict(endpoints)
print(est_exc)
'''
'''
model = keras.Sequential(
    [
        layers.Dense(2, activation='linear', name='endpoints'),
        layers.Dense(32, activation='relu', name='layer1'),
        layers.Dense(64, activation='relu', name='layer2'),
        layers.Dense(128, activation='relu', name='layer3'),
        layers.Dense(64, activation='relu', name='layer4'),
        layers.Dense(32, activation='relu', name='layer5'),
        layers.Dense(4, activation='linear', name='excursions')
    ]
)
model.compile(loss='mse', optimizer='adam')
model.fit(endpoints, excursions, epochs=6000, verbose=1)
'''

LL = '/media/tim/Chonky/Programming/VS/movingFromLaptop/softModel/3_5cmModel.json'
link = '/media/tim/Chonky/Programming/VS/movingFromLaptop/softModel/3_5cmModel_weights.h5'

#model.save(link)
#with open(LL, "w") as json_file:
    #json_file.write(model.to_json())
#model.save_weights(link)
'''
with open(LL) as saved_json:
    model = tf.keras.models.model_from_json(saved_json.read())
model.build()
model.summary()
model.load_weights(link)
'''
model = keras.models.load_model(link)

y = model.predict(endpoints)
print(y)