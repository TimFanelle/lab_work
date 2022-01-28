#imports
import RTBridge as rtb
import numpy as np
from numpy import matlib
from scipy import signal
from sklearn.neural_network import MLPRegressor
from matplotlib import pyplot as plt
#import os
#from copy import deepcopy
#import math

###### Called in Main ######

def babbling_func(listenAt, sendAt, babbling_min=3, timestep=1000):
    np.random.seed(0)

    #set constants
    sim_time = babbling_min*60.0
    run_samples = int(np.round(sim_time/timestep))
    
    max_in = 0.9
    min_in = 0.1

    pass_chance = timestep

    #generate all motor activations for babbling
    motor1_act = systemID_input_gen_func(signal_dur_in_sec=sim_time, pass_chance=pass_chance, max_in=max_in, min_in=min_in, timestep=timestep)
    motor2_act = systemID_input_gen_func(signal_dur_in_sec=sim_time, pass_chance=pass_chance, max_in=max_in, min_in=min_in, timestep=timestep)
    motor3_act = systemID_input_gen_func(signal_dur_in_sec=sim_time, pass_chance=pass_chance, max_in=max_in, min_in=min_in, timestep=timestep)
    
    #collect activations into a single list
    babbling_act = np.transpose(np.concatenate([[motor1_act], [motor2_act], [motor3_act]], axis=0))

    #show kinematics and activations
    kin_act_show_func(activations=babbling_act, timestep=timestep)

    #run activations
    [babbling_kin, babbling_act] = run_act_func(babbling_act, model_ver=0, timestep=timestep, listenAt=listenAt, sendAt=sendAt)

    return babbling_kin[1000:,::], babbling_act[1000:,:]

def inv_mapping_func(kinematics, activations, early_stopping=False, **kwargs):
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
            hidden_layer_sizes=13, activations="logistic",
            verbose = True, warm_start=True,
            early_stopping=early_stopping)

    #fit model
    model.fit(kin_train, act_train)

    #test run the model
    est_act = model.predict(kinematics)

    return model

###### One layer down #######

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

def kin_act_show_func(vs_time=False, timestep=0.005, **kwargs):
    sample_n_kin = 0
    sample_n_act = 0

    if("kinematics" in kwargs):
        kinematics = kwargs["kinematics"]
        sample_n_kin = kinematics.shape[0]
    if("activations" in kwargs):
        activations = kwargs['activations']
        sample_n_act = activations.shape[0]
    if not (("kinematics" in kwargs) or ("activations" in kwargs)):
        raise NameError("Please pass in either kinematics or activations")
    if (sample_n_kin != 0) and (sample_n_act != 0) and (sample_n_kin != sample_n_act):
        raise ValueError("Kinematics and activations lengths must be equivalent")
    else:
        num_samples = np.max([sample_n_kin, sample_n_act])
        if vs_time:
            x = np.linspace(0, timestep*num_samples, num_samples)
        else:
            x = range(num_samples)
    if ("kinematics" in kwargs):
        plt.figure()
        plt.subplot(6, 1, 1)
        plt.plot(x, kinematics[:,0])
        plt.ylabel('q0 (rads)')
        plt.subplot(6, 1, 2)
        plt.plot(x, kinematics[:,1])
        plt.ylabel('q0 dot (rads/s)')
        plt.subplot(6, 1, 3)
        plt.plot(x, kinematics[:,2])
        plt.ylabel('q0 double dot (rads/s^2)')
        plt.subplot(6, 1, 4)
        plt.plot(x, kinematics[:,3])
        plt.ylabel('q1 (rads)')
        plt.subplot(6, 1, 5)
        plt.plot(x, kinematics[:,4])
        plt.ylabel('q1 dot (rads/s)')
        plt.subplot(6, 1, 6)
        plt.plot(x, kinematics[:,5])
        plt.ylabel('q1 double dot (rads/s^2)')
        plt.xlabel('motor 1 activation values')
    if ("activations" in kwargs):
        plt.figure()
        plt.subplot(3, 1, 1)
        plt.plot(x, activations[:,0])
        plt.ylabel('motor 1 activation values')
        plt.subplot(3, 1, 2)
        plt.plot(x, activations[:,1])
        plt.ylabel('motor 1 activation values')
        plt.subplot(3, 1, 3)
        plt.plot(x, activations[:,2])
        plt.ylabel('motor 1 activation values')
        plt.xlabel('Sample #')
        plt.show(block=True) 

def run_act_func(activations, listenAt, sendAt, model_ver=0, timestep=1000):
    #define variables
    num_task_samples = activations.shape[0]
    real_attempt_excurs = np.zeros((num_task_samples, 3)) #one for each tendon we are measuring the excursion of
    real_attempt_act = np.zeros((num_task_samples, 3))
    
    #connect to RTB
    sof = rtb.BridgeSetup(sendAt, listenAt, rtb.setups.hand_3_3)
    sof.startConnection()

    #loop through activations and keep track of excursions to calculate velocity and acceloration
    for ijk in range(num_task_samples):
        postureExcursions = []
        np.concatenate(real_attempt_act, activations[ijk])
        degreeSet = sof.sendAndReceive(activations[ijk], timestep)
        for p in range(len(degreeSet)):
            degreeSet[p] = round(degrees2excurs(degreeSet[p], 6), 4)
        np.concatenate(real_attempt_excurs, np.array(degreeSet))

    #kill connection to RTB
    zeroSet = np.zeros((1, activations.shape[1]))
    _ = sof.sendAndReceive(zeroSet, 0.0005)
	#in python this happens automatically

    #convert from excursions to kinematics
    real_attempt_kin = excurs2kin_func(real_attempt_excurs[:,0], real_attempt_excurs[:,1], real_attempt_excurs[:,2], timestep = timestep)

    return real_attempt_kin, real_attempt_act


###### Another Layer down ######

def excurs2kin_func(t0, t1, t2, timestep=1000):
    return 0

def degrees2excurs(degrees, diameterInMM=6):
	return (np.pi*diameterInMM)*(degrees/360)
