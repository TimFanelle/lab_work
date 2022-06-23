#imports
import RTBridge as rtb
import numpy as np
from sklearn.neural_network import MLPRegressor
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import load_model


##### main functions #####

def babbling_func(babbling_min=3, timestep=0.5, **kwargs):
    
    np.random.seed(0)
    
    #set constants
    sim_time = babbling_min*60.0
    run_samples = int(np.round(sim_time/timestep))

    max_in = 0.95
    min_in = 0.05

    pass_chance = timestep

    #generate motor activations
    motor1_act = systemID_input_gen_func(signal_dur_in_sec=sim_time, pass_chance=pass_chance, max_in=max_in, min_in=min_in, timestep=timestep)
    motor2_act = systemID_input_gen_func(signal_dur_in_sec=sim_time, pass_chance=pass_chance, max_in=max_in, min_in=min_in, timestep=timestep)
    motor3_act = systemID_input_gen_func(signal_dur_in_sec=sim_time, pass_chance=pass_chance, max_in=max_in, min_in=min_in, timestep=timestep)

    #collect activations into a single list
    babbling_act = np.transpose(np.concatenate([[motor1_act],[motor2_act], [motor3_act]], axis=0))

    #skipping showing kinematics and activations

    # run activations
    if ("listenAt" in kwargs) and ("sendAt" in kwargs):
        [babbling_kin, babbling_act] = run_act_func(babbling_act, timestep=int(np.round(timestep*1000)), listenAt=kwargs['listenAt'], sendAt=kwargs['sendAt'], babbling=True)
        babbling_kin = babbling_kin[6:,:] 
        babbling_act = babbling_act[6:,:]
    else:
        if not (("listenAt" in kwargs)) or ("sendAt" in kwargs)):
            [babbling_kin, babbling_act] = run_act_func(babbling_act, timestep=int(np.round(timestep*1000)), babbling=True)
            babbling_kin = babbling_kin[1000:,:] 
            babbling_act = babbling_act[1000:,:]
        else:
            raise ValueError("Both listenAt and sendAt are needed to use RTB")
    
    return babbling_kin, babbling_act

def inverse_mapping_func(kinematics, activations, early_stopping=False, **kwargs):
    #define constants
    num_samples=activations.shape[0]
    train_ratio = 1
    kin_train = kinematics[:int(np.round(train_ratio*num_samples)),:]
    kin_test = kinematics[int(np.round(train_ratio*num_samples))+1:,:]
    act_train = activations[:int(np.round(train_ratio*num_samples)),:]
    act_test = activations[int(np.round(train_ratio*num_samples))+1:,:]
    num_samples_test = act_test.shape[0]

    #set model to prior model if available, else do a regression to map kinematics to activations
    print("training the model")
    if("prior_model" in kwargs):
        model=kwargs["prior_model"]
    else:
        model = MLPRegressor(
            hidden_layer_sizes=13, activation="logistic",
            verbose = True, warm_start=True,
            early_stopping=early_stopping)

    #fit model
    model.fit(kin_train, act_train)

    #test run the model
    est_act = model.predict(kinematics)

    return model

def initial_learning_func(model, babbling_kin, babbling_act, num_refinements=10, timestep=0.5, **kwargs):
    cum_kin = babbling_kin
    cum_act = babbling_act
    attempt_kin = findKin_func(attempt_length=50, num_cycles=7, timestep=timestep)
    est_attempt_act = estimate_act_func(model=model, desired_kin=attempt_kin, timestep=timestep)

    if ("listenAt" in kwargs) and ("sendAt" in kwargs):
        [real_attempt_kin, real_attempt_act] = run_act_func(est_attempt_act, timestep=int(np.round(timestep*1000)), listenAt=kwargs['listenAt'], sendAt=kwargs['sendAt'])
    else:
        #this is dumb and shouldn't come up
        real_attempt_kin = np.array([0,0,0])
        real_attempt_act = np.array([0,0,0])
    
    error_0 = np.array([error_calc_func(updated_kin[:,0], real_attempt_kin[:,0])])
    error_1 = np.array([error_calc_func(updated_kin[:,3], real_attempt_kin[:,3])])
    error_2 = np.array([error_calc_func(updated_kin[:,6], real_attempt_kin[:,6])])

    avg_err = np.average([error_0, error_1, error_2])

    for ijk in range(num_refinements):
        print("Refinement Number: ", ijk+1)
        [cum_kin, cum_act] = concat_data_func(cum_kin, cum_act, real_attempt_kin, real_attempt_act)
        model = inv_mapping_func(kinematics=cum_kin, activations=cum_act, prior_model=model)
        est_attempt_Act, updated_kin = estimate_act_fun(model=model, desired_kin=attempt_kin)
        if ("listenAt" in kwargs) and ("sendAt" in kwargs):
            [real_attempt_kin, real_attempt_act] = run_act_func(est_attempt_act, timestep=int(np.round(timestep*1000)), listenAt=kwargs['listenAt'], sendAt=kwargs['sendAt'])
        else:
            #this is dumb and shouldn't come up
            real_attempt_kin = np.array([0,0,0])
            real_attempt_act = np.array([0,0,0])
        
        error_0 = np.append(error_0, error_calc_func(updated_kin[:,0], real_attempt_kin[:,0]))
        error_1 = np.append(error_1, error_calc_func(updated_kin[:,3], real_attempt_kin[:,3]))
        error_2 = np.append(error_2, error_calc_func(updated_kin[:,6], real_attempt_kin[:,6]))
    
    errors = np.concatenate([[error_0], [error_1], [error_2]], axis=0)

    return model, errors, cum_kin, cum_act

def moveCycling(model, listenAt, sendAt, timestep=0.5, attempt_length=30, num_cycles=7):
    attempt_kin = findKin_func(attempt_length=attempt_length, num_cycles=num_cycles, timestep=timestep)
    estAttemptAct, updated_kin = estimate_act_fun(model=model, desired_kin=attempt_kin, timestep=timestep)

    [real_attempt_kin, real_attempt_act] = run_act_func(estAttemptAct, listenAt=listenAt, sendAt=sendAt, timestep=int(np.round(timestep*1000)))

    return real_attempt_kin, real_attempt_act

##### Main support functions #####

def systemID_input_gen_func(signal_dur_in_sec, pass_chance, max_in, min_in, timestep):
    #variable definitions
    num_samples = int(np.round(signal_dur_in_sec/timestep))
    samples = np.linspace(0, signal_dur_in_sec, num_samples)
    gen_input = np.zeros(num_samples,)*min_in

    for ijk in range(1, num_samples):
        rand_pass = np.random.uniform(0,1,1)
        if rand_pass < pass_chance:
            gen_input[ijk] = ((max_in-min_in)*np.random.uniform(0,1,1))+min_in
        else:
            gen_input[ijk] = gen_input[ijk-1]
        
    return gen_input

def run_act_func(activations, timestep=1000, **kwargs):
    #define variables
    num_task_samples = activations.shape[0]
    real_attempt_excurs = np.zeros((num_task_samples, 4)) #one for each tendon we are measuring the excursion of
    real_attempt_act = np.zeros((num_task_samples, 4))
    
    if ("listenAt" in kwargs) and ("sendAt" in kwargs):
        #connect to RTB
        sof = rtb.BridgeSetup(sendAt, listenAt, rtb.setups.hand_4_4)
        sof.startConnection()

        #loop through activations and keep track of excursions to calculate velocity and acceloration
        for ijk in range(num_task_samples):
            postureExcursions = []
            if ("babbling" in kwargs):
                adjusted_act = adjust_act_func(activations[ijk])
                real_attempt_act[ijk,:] = adjusted_act #activations[ijk]
            else:
                real_attempt_act[ijk,:] = activations[ijk,:]

            #a = (((100+45)*np.random.uniform(0,1,1))-45)[0]
            #b =(((100+45)*np.random.uniform(0,1,1))-45)[0]
            #c=(((100+45)*np.random.uniform(0,1,1))-45)[0]
            #degreeSet = [a, b, c]
            
            degreeSet = sof.sendAndReceive(real_attempt_act[ijk,:], timestep)
            #print(degreeSet)
            for p in range(len(degreeSet)):
                degreeSet[p] = np.round(degrees2excurs(degreeSet[p], 6), 4)
            real_attempt_excurs[ijk,:] = np.array(degreeSet)

        #kill connection to RTB
        zeroSet = np.zeros(activations.shape[1])
        _ = sof.sendAndReceive(zeroSet, 2)
        #in python this happens automatically
    else:
        #something else will happen here
    #convert from excursions to kinematics
    real_attempt_kin = excurs2kin_func(real_attempt_excurs[:,0], real_attempt_excurs[:,1], real_attempt_excurs[:,2], real_attempt_excurs[:,3], timestep = timestep)

    return real_attempt_kin, real_attempt_act

def findKin_func(attempt_length=10, num_cycles=7, timestep=1):
    #TODO: rewrite this to do what I need it to
    #this will assume sine on Q0 and cosine on Q1

    num_attempt_samples = int(np.round(attempt_length/timestep))
    q0 = np.zeros(num_attempt_samples)
    q1 = np.zeros(num_attempt_samples)

    for ijk in range(num_attempt_samples):
        q0[ijk] = (np.pi/4) * np.sin(num_cycles*(2*np.pi*ijk/num_attempt_samples))
        q1[ijk] = -1*(np.pi/2)*((0.5*np.cos(num_cycles*(2*np.pi*ijk/num_attempt_samples))-0.5))
    
    attempt_kin = angle2endpoint(q0, q1)

    return attempt_kin

def estimate_act_fun(model, desired_kin, timestep=1):
    #TODO: rewrite this to use geometic and moment arm matrices
    
    link = '/media/tim/Chonky/Programming/VS/liveFinger/DARPA/modeling/singleFingTest.h5'
    if (not os.path.isfile(link)):
        print("Model not found, creating one from internal data")
        r = np.array(
        [  #joints down by tendons across
        [6,7,2],
        [4,12,6]
        ])
        lengths = np.array([36, 36])

        trainNetwork(link, r, lengths)
    
    endp2kin = keras.models.load_model(link)
    updated_kin = endp2kin.predict(desired_kin)
    updated_kin = excurs2kin_func(updated_kin[:,0], updated_kin[:,1], updated_kin[:,2], timestep=timestep)
    est_act = model.predict(updated_kin)

    return est_act, updated_kin

def concat_data_func(cum_kin, cum_act, kin, act, throw_percentage=0.2):
    size_incoming_data = kin.shape[0]
    samples_to_throw = int(np.round(throw_percentage*size_incoming_data))
    cum_kin = np.concatenate([cum_kin, kin[samples_to_throw:,:]])
    cum_act = np.concatenate([cum_act, act[samples_to_throw:,:]])
    return cum_kin, cum_act

def error_calc_func(input1, input2):
    error = np.mean(np.abs(input1-input2))
    return error

##### Support Support Functions #####

def excurs2kin_func(t0, t1, t2, timestep=1000):
    kinematics = np.transpose(np.concatenate((
	[[t0], [np.gradient(t0)/timestep], [np.gradient(np.gradient(t0)/timestep)/timestep], 
	[t1], [np.gradient(t1)/timestep], [np.gradient(np.gradient(t1)/timestep)/timestep], 
	[t2], [np.gradient(t2)/timestep], [np.gradient(np.gradient(t2)/timestep)/timestep]]), axis=0))
    return kinematics

def degrees2excurs(degrees, diameterInMM=6):
	return (np.pi*diameterInMM)*(degrees/360)

def adjust_act_func(activations):
    adjust_act = []
    adjustment = 0.5
    max_index = np.where(activations == max(activations))
    for i in range(len(activations)):
        if i == max_index:
            adjust_act.append(activations[i])
        else:
            adjust_act.append(adjustment*activations[i])
    return np.array(adjust_act)\

def angle2endpoint(q0, q1, l1=3.6, l2=3.6):
    try:
        xy = np.zeros((q0.shape[0],2))

        for ijk in range(q0.shape[0]):
            xy[ijk][0] = (l1*np.cos(q0[ijk]))+(l2*np.cos(q0[ijk]+q1[ijk]))
            xy[ijk][1] = (l1*np.sin(q0[ijk]))+(l2*np.sin(q0[ijk]+q1[ijk]))
    except AttributeError:
        xy = [0,0]
        q0 = np.radians(q0)
        q1 = np.radians(q1)
        xy[0] = (l1*np.cos(q0))+(l2*np.cos(q0+q1))
        xy[1] = (l1*np.sin(q0))+(l2*np.sin(q0+q1))
    return xy

def trainNetwork(link, r, lengths, q0Range, q1Range):
    excursions = []
    xy = []
    for i in range(q0Range[0], q0Range[1]):
        for j in range(q1Range[0], q1Range[1]):
            excursions.append(np.matmul(-1*np.transpose(r),np.array([np.radians(i),np.radians(j)])))
            xy.append(angle2endpoint(i,j,lengths[0], lengths[1]))
    excursions = np.array(excursions)
    xy = np.array(xy)
    #print(excursions)
    #print(xy)

    model = keras.Sequential(
    [
        layers.Dense(2, activation='linear', name='endpoints'),
        layers.Dense(32, activation='relu', name='layer1'),
        layers.Dense(64, activation='relu', name='layer2'),
        #layers.Dense(128, activation='relu', name='layer3'),
        layers.Dense(64, activation='relu', name='layer4'),
        layers.Dense(32, activation='relu', name='layer5'),
        layers.Dense(3, activation='linear', name='excursions')
    ]
    )
    model.compile(loss='mse', optimizer='adam')
    model.fit(xy, excursions, epochs=200, verbose=1)
    #print(model.summary())
    model.save(link)