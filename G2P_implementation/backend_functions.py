#imports
from mujoco_py import load_model_from_path, MjSim, MjViewer
import numpy as np
from numpy import matlib
from scipy import signal
from sklearn.neural_network import MLPRegressor
from matplotlib import pyplot as plt
#import pickle
import os
from copy import deepcopy
from mujoco_py.generated import const
import math

##### Called in Main #####

def babbling_fcn(simulation_minutes=5, xmlpath=""):
    '''
    This function generates babbling kinematics and preceding activation sets 
    for the length of time given and the model at the path given
    '''
    np.random.seed(0)

    #grab model from provided path
    xmlpath_end = "/media/tim/Chonky/Programming/VS/movingFromLaptop/Models/"+xmlpath+".xml" #/media/tim/Chonky/Programming/VS/movingFromLaptop/Models
    model = load_model_from_path(xmlpath_end)
    #define simulation using model
    sim = MjSim(model)
    sim_state = sim.get_state()

    #set constants
    control_vec_len = sim.data.ctrl.__len__()
    sim_time = simulation_minutes*60.0
    dt = 0.005
    run_samples = int(np.round(sim_time/dt))
    max_in = 1
    min_in = 0
    pass_chance = dt

    sim.set_state(sim_state)
    #Generate all motor activations for babbling session
    motor1_act = systemID_input_gen_fcn(signal_duration_in_seconds=sim_time, pass_chance=pass_chance, max_in=max_in, min_in=min_in, timestep=dt)
    motor2_act = systemID_input_gen_fcn(signal_duration_in_seconds=sim_time, pass_chance=pass_chance, max_in=max_in, min_in=min_in, timestep=dt)
    motor3_act = systemID_input_gen_fcn(signal_duration_in_seconds=sim_time, pass_chance=pass_chance, max_in=max_in, min_in=min_in, timestep=dt)
    
    #Collect all activations into a single list
    babbling_act = np.transpose(np.concatenate([[motor1_act],[motor2_act],[motor3_act]],axis=0))

    #Show kinematics and activations
    kin_act_show_func(activations=babbling_act)
    
    #Reset simulation state and run activations
    sim.set_state(sim_state)
    [babbling_kin, babbling_act, chassis_pos] = run_act_func(babbling_act, model_ver=0, timestep=0.005, Mj_render=False, xmlpath=xmlpath)
    #import pdb; pdb.set_trace()
    #return kinematics and prior activation set
    return babbling_kin[1000:,:],babbling_act[1000:,:]

def inv_mapping_func(kinematics, activations, early_stopping=False, **kwargs):
    '''
    This function takes in model kinematics and activations and does an inverse mapping to
    create a model that relates kinematics to activations
    '''
    
    #define constants
    num_samples=activations.shape[0]
    train_ratio = 1
    kin_train = kinematics[:int(np.round(train_ratio*num_samples)),:]
    kin_test = kinematics[int(np.round(train_ratio*num_samples))+1:,:]
    act_train = activations[:int(np.round(train_ratio*num_samples)),:]
    act_test = activations[int(np.round(train_ratio*num_samples))+1:,:]
    num_samples_test = act_test.shape[0]

    #set model to previously trained mapped if available, else do a regression to map kinematics to activations
    print("Training the model")
    if("prior_model" in kwargs):
        model=kwargs["prior_model"]
    else:
        model = MLPRegressor(
            hidden_layer_sizes=11, activation="logistic",  #original is 15 and "logistic"
            verbose=True, warm_start=True, 
            early_stopping=early_stopping)

    #fit model
    model.fit(kin_train, act_train)
    
    #test run the model
    est_act = model.predict(kinematics)

    #return model
    return model

def learn2move_func(model, cum_kin, cum_act, reward_thresh=6, refinement=False, Mj_render=False, xmlpath=""):
    '''
    This function takes in a model, kinematics, activations, and a reward threshold
    then loops until generated features result in a reward above the given threshold
    and returns the best reward and all rewards
    '''
    #define core variables
    prev_reward = np.array([0])
    best_reward_so_far = prev_reward
    best_model = model
    all_rewards = []
    exploitation_run_no = 0
    new_features = gen_feat_func(prev_reward=prev_reward, reward_thresh=reward_thresh,best_reward_so_far=best_reward_so_far, feat_vec_length=10)
    best_features_so_far = new_features

    #repeatedly generate and test features until the best reward is above the threshold 16 times
    i = 0
    while exploitation_run_no<=15:
        #check if reward is above threshold
        if best_reward_so_far>reward_thresh:
            exploitation_run_no+=1
        #generate and run new features based on the previous best features
        new_features = gen_feat_func(reward_thresh=reward_thresh, best_reward_so_far=best_reward_so_far, best_features_so_far=best_features_so_far)
        [prev_reward, attempt_kin, est_attempt_act, real_attempt_kin, real_attempt_act] = feat2run_attempt_func(features=new_features, model=model, feat_show=False, model_vers=1, xmlpath=xmlpath)
        #concatenate kinematics, activations, and rewards
        [cum_kin, cum_act] = concat_data_func(cum_kin, cum_act, real_attempt_kin, real_attempt_act, throw_percentage=0.2)
        all_rewards = np.append(all_rewards, prev_reward)
        
        #update best_reward_so_far as necessary
        if prev_reward>best_reward_so_far:
            best_reward_so_far = prev_reward
            best_features_so_far = new_features
            #save copy of best model
            best_model = deepcopy(model)
        
        #update model each loop
        if refinement:
            model = inv_mapping_func(cum_kin, cum_act, prior_model=model)
        print("Iteration #: ", i)
        i+=1
        print("best reward so far: ", best_reward_so_far)

    input("Learning to walk completed, press any key to continue")

    #One final run to get best outputs and attempts
    [prev_reward_best, attempt_kin_best, est_attempt_act_best, real_attempt_kinematics_best, real_attempt_act_best] = feat2run_attempt_func(features=best_features_so_far, model=best_model,feat_show=True, Mj_render=Mj_render)
    kin_act_show_func(vs_time=True, kinematics=real_attempt_kin)
    print("All rewards: ", all_rewards)
    print("Previous reward best: ", prev_reward_best)

    #return rewards
    return best_reward_so_far, all_rewards

def inAirTraining_func(model, babbling_kinematics, babbling_activations, num_refinements=10, Mj_render=False, xmlpath_end=""):
    Mj_render_last_run = False
    model_ver = 0
    
    cum_kin = babbling_kinematics
    cum_act = babbling_activations

    attempt_kin = makeCircle_func(attempt_length=10, num_circles=6) #createSinCosKin_func(attempt_length=10, number_of_cycles=7)
    estAttemptAct = estimate_act_func(model=model, desired_kin=attempt_kin)
    
    if (num_refinements == 0) and (Mj_render == True):
        Mj_render_last_run = True
    
    [real_attempt_kin, real_attempt_act, chassis_pos] = run_act_func(estAttemptAct, model_ver = model_ver, Mj_render=Mj_render_last_run, xmlpath=xmlpath_end)
    #error in position of each joint
    error_0 = np.array([error_calc_func(attempt_kin[:,0], real_attempt_kin[:,0])])
    error_1 = np.array([error_calc_func(attempt_kin[:,3], real_attempt_kin[:,3])])
    avg_err = (error_0+error_1)/2

    for ijk in range(num_refinements):
        if (ijk+1 == num_refinements) and (Mj_render==True):
            Mj_render_last_run = True
        
        print("Refinement number: ", ijk+1)
        [cum_kin, cum_act] = concat_data_func(cum_kin, cum_act, real_attempt_kin, real_attempt_act)
        model = inv_mapping_func(kinematics=cum_kin, activations=cum_act, prior_model=model)
        est_attempt_act = estimate_act_func(model=model, desired_kin=attempt_kin)
        [real_attempt_kin, real_attempt_act, chassis_pos] = run_act_func(est_attempt_act, model_ver=model_ver, Mj_render=Mj_render_last_run, xmlpath=xmlpath_end)
        error_0 = np.append(error_0, error_calc_func(attempt_kin[:,0], real_attempt_kin[:,0]))
        error_1 = np.append(error_1, error_calc_func(attempt_kin[:,3], real_attempt_kin[:,3]))
        avg_err = np.append(avg_err, (error_0[-1]+error_1[-1])/2)

    #idk, some plotting stuff

    errors = np.concatenate([[error_0], [error_1]], axis=0)

    return model, errors, cum_kin, cum_act

##### Called here layer 1 #####

def systemID_input_gen_fcn(signal_duration_in_seconds, pass_chance, max_in, min_in,timestep):
    '''
    This function takes in the simulation time in seconds, the pass chance, the maximum output,
    the minimum output, and the timestep between activations then outputs a uniformly distributed
    set of activations that the given duration split into one activation for each time step
    '''
    #variable defintions
    num_samples = int(np.round(signal_duration_in_seconds/timestep))
    samples = np.linspace(0, signal_duration_in_seconds,num_samples)
    gen_input = np.zeros(num_samples,)*min_in

    #for the range of 1:num_samples generate a random number and if it is greater than pass chance 
    #then use the same activation as the previous timestep, else generate a new activation
    for ijk in range(1, num_samples):
        rand_pass = np.random.uniform(0,1,1)
        if rand_pass < pass_chance:
            gen_input[ijk] = ((max_in-min_in)*np.random.uniform(0,1,1))+min_in
        else:
            gen_input[ijk] = gen_input[ijk-1]
    #return generated activations
    return gen_input

def kin_act_show_func(vs_time=False, timestep=0.005, **kwargs):
    '''
    This function plots kinematics and activations
    '''
    sample_n_kin = 0
    sample_n_act = 0

    if("kinematics" in kwargs):
        kinematics = kwargs["kinematics"]
        sample_n_kin = kinematics.shape(0)
    if("activations" in kwargs):
        activations = kwargs["activations"]
        sample_n_act = activations.shape[0]
    if not (("kinematics" in kwargs) or ("activations" in kwargs)):
        raise NameError("Please pass either kinematics or activations")
    if (sample_n_kin != 0) and (sample_n_act != 0) and (sample_n_act != sample_n_kin):
        raise ValueError("Kinematics and Activations must be the same length and not equal to zero")
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

def run_act_func(est_activations, model_ver=0, timestep=0.005, Mj_render=False, xmlpath=""):
    '''
    This function runs the given activations generated from running the inverse map on
    the kinematics of the desired task
    '''

    #variable definitions
    xmlpath_end = "/media/tim/Chonky/Programming/VS/movingFromLaptop/Models/"+xmlpath+".xml" #/media/tim/Chonky/Programming/VS/movingFromLaptop/Models
    model = load_model_from_path(xmlpath_end)
    sim = MjSim(model)

    sim_state = sim.get_state()
    control_vec_len = sim.data.ctrl.__len__()
    num_task_samples = est_activations.shape[0]
    
    if Mj_render:
        viewer = MjViewer(sim)
    
    #print("Ctrl Vel Len: "+ str(control_vec_len))

    #positions and activations
    real_attempt_pos = np.zeros((num_task_samples, 2))
    real_attempt_act = np.zeros((num_task_samples, 3))
    chassis_pos = np.zeros(num_task_samples,)

    sim.set_state(sim_state) #reset state

    for ijk in range(num_task_samples):
        sim.data.ctrl[:] = est_activations[ijk,:] #set controller to generated activations
        sim.step() #advance simulation by one step
        cur_pos_array = sim.data.qpos[-2:] #get current position (joint angles?)
        
        #TODO: check this
        chassis_pos[ijk] = sim.data.get_geom_xpos("Endpoint")[0] #acquire position of endpoint
        
        #save results
        real_attempt_pos[ijk,:] = cur_pos_array
        real_attempt_act[ijk,:] = sim.data.ctrl

        #render if desired
        if Mj_render:
            viewer.render()
    
    #convert position to kinematics and concatenate into a singular list
    real_attempt_kin = pos2kin_func(real_attempt_pos[:,0], real_attempt_pos[:,1],timestep=0.005)
    
    #return kinematics, activations, and chassis position(endpoint position)
    return real_attempt_kin, real_attempt_act, chassis_pos

def gen_feat_func(reward_thresh, best_reward_so_far, **kwargs):
    '''
    Takes in desired reward_threshold and the best reward so far and generates
    network features based on that
    '''
    #TODO: determine what the feature min and max should be
    feat_min = 0.4 #originally 0.4
    feat_max = 0.9 #originally 0.9

    #define best_features_so_far based on given input
    if("best_features_so_far" in kwargs):
        best_features_so_far = kwargs["best_features_so_far"]
    elif ("feat_vec_length" in kwargs):
        best_features_so_far = np.random.uniform(feat_min, feat_max, kwargs["feat_vec_length"])
    else:
        raise NameError("Either best_features_so_far or feat_vec_length is needed")
    
    #generate new random features if the reward is below the threshold
    if best_reward_so_far < reward_thresh:
        new_features = np.random.uniform(feat_min, feat_max, best_features_so_far.shape[0])
    else: # or slightly modify the current features based on the best reward if the reward is above threshold
        sigma = np.max([(12-best_reward_so_far)/100, 0.01]) 
        new_features = np.zeros(best_features_so_far.shape[0],)
        for ijk in range(best_features_so_far.shape[0]):
            new_features[ijk] = np.random.normal(best_features_so_far[ijk], sigma) #generate features randomly between the current best and sigma for each feature
        
        #take the max of the minimums and the minimum of the maximums
        new_features = np.maximum(new_features, feat_min*np.ones(best_features_so_far.shape[0],))
        new_features = np.minimum(new_features, feat_max*np.ones(best_features_so_far.shape[0],))
    
    #return the new features
    return new_features

def feat2run_attempt_func(features, model, feat_show=False, Mj_render=False, model_vers=1, xmlpath=""):
    [q0_filtered, q1_filtered] = feat2pos_func(features, show=feat_show)
    step_kin = pos2kin_func(q0_filtered, q1_filtered, timestep=0.005)
    attempt_kin = step2attempt_kin_func(step_kin=step_kin)
    est_attempt_act = estimate_act_func(model=model, desired_kin=attempt_kin)
    [real_attempt_kin, real_attempt_act, chassis_pos] = \
        run_act_func(est_attempt_act, model_ver=0, Mj_render=Mj_render, xmlpath=xmlpath)
    prev_reward = chassis_pos[-1]
    return prev_reward, attempt_kin, est_attempt_act, real_attempt_kin, real_attempt_act

def concat_data_func(cum_kin, cum_act, kin, act, throw_percentage=0.2):
    size_incoming_data = kin.shape[0]
    samples_to_throw = int(np.round(throw_percentage*size_incoming_data))
    cum_kin = np.concatenate([cum_kin, kin[samples_to_throw:,:]])
    cum_act = np.concatenate([cum_act, act[samples_to_throw:,:]])
    return cum_kin, cum_act

def createSinCosKin_func(attempt_length=10, number_of_cycles=7, timestep = 0.005):
    num_attempt_samples = int(np.round(attempt_length/timestep))
    
    q0=np.zeros(num_attempt_samples)
    q1=np.zeros(num_attempt_samples)

    for ijk in range(num_attempt_samples):
        q0[ijk] = (np.pi/3)*np.sin(number_of_cycles*(2*np.pi*ijk/num_attempt_samples))
        q1[ijk] = -1*(np.pi/2)*((-1*np.cos(number_of_cycles*(2*np.pi*ijk/num_attempt_samples))+1)/2)

    attempt_kin = pos2kin_func(q0, q1, timestep)

    return attempt_kin

def error_calc_func(input1, input2):
    error = np.mean(np.abs(input1-input2))
    return error

def makeCircle_func(attempt_length=10, num_circles=6, timestep=0.005):
    num_attempt_samples = int(np.round(attempt_length/timestep))

    q0=np.zeros(num_attempt_samples)
    q1=np.zeros(num_attempt_samples)

    for ijk in range(num_attempt_samples):
        q0[ijk] = (np.pi/9)*np.sin(num_circles*(2*np.pi*ijk/num_attempt_samples))
        q1[ijk] = (-0.5*(np.cos(num_circles*(2*np.pi*ijk/num_attempt_samples))))+0.5
    
    print(q1)
    attempt_kin = pos2kin_func(q0, q1, timestep)

    return attempt_kin
        

##### Called here layer 2 #####

def pos2kin_func(q0, q1, timestep = 0.005):
    kinematics = np.transpose(np.concatenate(([[q0],[np.gradient(q0)/timestep],[np.gradient(np.gradient(q0)/timestep)/timestep],[q1],[np.gradient(q1)/timestep],[np.gradient(np.gradient(q1)/timestep)/timestep]]), axis=0))
    return kinematics

def step2attempt_kin_func(step_kin, number_of_steps_in_an_attempt=10):
    attempt_kin = np.matlib.repmat(step_kin, number_of_steps_in_an_attempt, 1)
    return(attempt_kin)

def estimate_act_func(model, desired_kin):
    #print("running the model")
    est_act = model.predict(desired_kin)
    return est_act

def feat2pos_func(features, timestep=0.005, cycle_duration_in_seconds=1.3, show=False):
    num_feat = features.shape[0]
    each_feat_len = int(np.round((cycle_duration_in_seconds/num_feat)/timestep))
    feat_angles = np.linspace(0, 2*np.pi*(num_feat/(num_feat+1)), num_feat)
    
    q0_raw = features*np.sin(feat_angles)
    q1_raw = features*np.cos(feat_angles)
    
    q0_scaled = (q0_raw*np.pi/3)
    q1_scaled = -1*((-1*q1_raw+1)/2)*(np.pi/2)

    q0_scaled_extended = np.append(q0_scaled, q0_scaled[0])
    q1_scaled_extended = np.append(q1_scaled, q1_scaled[0])
    q0_scaled_extended_long = np.array([])
    q1_scaled_extended_long = np.array([])

    for ijk in range(features.shape[0]):
        q0_scaled_extended_long = np.append(q0_scaled_extended_long, np.linspace(q0_scaled_extended[ijk], q0_scaled_extended[ijk+1], each_feat_len))
        q1_scaled_extended_long = np.append(q1_scaled_extended_long, np.linspace(q1_scaled_extended[ijk], q1_scaled_extended[ijk+1], each_feat_len))

    q0_scaled_extended_long_3 = np.concatenate([q0_scaled_extended_long[:-1], q0_scaled_extended_long[:-1], q0_scaled_extended_long[:-1]])
    q1_scaled_extended_long_3 = np.concatenate([q1_scaled_extended_long[:-1], q1_scaled_extended_long[:-1], q1_scaled_extended_long[:-1]])

    fir_filter_length = int(np.round(each_feat_len/(1)))
    b = np.ones(fir_filter_length, )/fir_filter_length
    a = 1

    q0_filtered_3 = signal.filtfilt(b,a,q0_scaled_extended_long_3)
    q1_filtered_3 = signal.filtfilt(b,a,q1_scaled_extended_long_3)

    q0_filtered = q0_filtered_3[q0_scaled_extended_long.shape[0]:2*q0_scaled_extended_long.shape[0]-1]
    q1_filtered = q1_filtered_3[q1_scaled_extended_long.shape[0]:2*q1_scaled_extended_long.shape[0]-1]

    if show:
        plt.figure()
        plt.scatter(q0_scaled, q1_scaled)
        plt.plot(q0_filtered, q1_filtered)
        plt.xlabel("q0")
        plt.ylabel("q1")
        plt.show(block=True)
    return q0_filtered, q1_filtered