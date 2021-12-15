import envs.lhc_env as myenv
from stable_baselines3 import TD3
import matplotlib.pyplot as plt
import pickle
import numpy as np
from time import time

''' This script tests the trained model on a given set of errors in the IR1 inntr triplet'''

def run_correction():

    # Nominal values of the inner triplet magnetic strengths
    KQX3B1 = -0.0234346323
    KQX2B1 = 0.04005580581
    KQX1B1 = -0.02385208
    KQX1B2 = 0.02385208
    KQX2B2 = -0.04005580581
    KQX3B2 = 0.0234346323

    nominal = [KQX3B1, KQX3B1, KQX2B1, KQX2B1,
             KQX1B1, KQX1B1, KQX1B2, KQX1B2,
             KQX2B2, KQX2B2, KQX3B2, KQX3B2]

    # Nominal values of beta functions at the Q1 locations in the 4 different IPs
    betaX_L1 = 1592.145382
    betaY_L1 = 1592.147517
    betaX_R1 = 1592.14538
    betaY_R1 = 1592.148218

    betaX_L2 = 56.63440158
    betaY_L2 = 56.63407795
    betaX_R2 = 56.63440274
    betaY_R2 = 56.63412429

    betaX_L5 = 1592.14537
    betaY_L5 = 1592.150588
    betaX_R5 = 1592.145368
    betaY_R5 = 1592.151234

    betaX_L8 = 158.4480108
    betaY_L8 = 158.4492521
    betaX_R8 = 158.4480109
    betaY_R8 = 158.4492562

    betas_B1 = [betaX_L1, betaX_R1, betaX_L2, betaX_R2,
                betaX_L5, betaX_R5, betaX_L8, betaX_R8,
                betaY_L1, betaY_R1, betaY_L2, betaY_R2,
                betaY_L5, betaY_R5, betaY_L8, betaY_R8,
                ]

    # Initialise environment
    env = myenv.LHCEnv()
    obs = env.reset()

    # Define errors and observation space
    errors = obs - nominal
    print("Nominal state:", nominal)
    print("State:", obs)
    print("Errors:", errors)
    print("Errors Rel:", errors/nominal)
    initial = obs
    obs_0 = obs

    # Estimate initial beta-beating via surrogate model
    modelfile = 'Ridge_surrogate_20k.pkl'
    estimator = pickle.load(open(modelfile, "rb"))

    # Estimated beta-beating before correction
    dbetas_0 = estimator.predict(np.reshape(initial, (1, -1))) / betas_B1 * 100
    beating_0 = abs(dbetas_0).mean()
    print("beta-beating before correction = %1.2f %%" % beating_0)

    # Manual correction based on compensating magnetic errors

    errors_avg = [0]*12

    for j in range(6):
        i = 2*j
        errors_avg[i], errors_avg[i+1] = (errors[i] + errors[i+1])/2.0, (errors[i] + errors[i+1])/2.0

    initial_corr = initial - errors_avg
    dbetas_manual = estimator.predict(np.reshape(initial_corr, (1, -1))) / betas_B1 * 100
    beating_manual = abs(dbetas_manual).mean()
    print("beta-beating after manual correction = %1.2f %%" % beating_manual)

    # Load RL model
    model = TD3.load("td3_LHC_136")

    done = False
    step = 0

    # Perform correction
    while not done:
        action, _states = model.predict(obs)
        #print("Action :", action)
        obs, rewards, done, info, dbetas = env.step(action)
        if done:
            beating = (-rewards+5)
            #print("Correction completed!")
            print("beta-beating after correction = %1.2f %%" % beating)
            #print("Correction =", action)
            #print("Final state =", obs)

    # Return beta-beating after correction
    obs_0_norm = (obs_0-nominal)/nominal*100
    obs_norm = (obs-nominal)/nominal*100

    '''
    plt.figure()
    plt.plot(obs_0_norm, 'o', label='Before correction')
    plt.plot(obs_norm, 'o', label='After Correction')
    plt.axhline(0, label='Perfect machine')
    plt.legend()
    plt.show()
    '''

    # Some post-processing of the action space
    rel_action = (np.array([action[0], action[0], action[1], action[1], action[2], action[2],
                  action[3], action[3], action[4], action[4], action[5], action[5]]))/np.array(nominal)

    action_vec = (np.array([action[0], action[0], action[1], action[1], action[2], action[2],
                  action[3], action[3], action[4], action[4], action[5], action[5]]))

    #print("Relative action:", rel_action)
    print("Absolute action:", action_vec)

    magnet_list = ["Q1XL1", "Q1XR1", "Q1XL2", "Q1XR2", "Q1XL5", "Q1XR5", "Q1XL8", "Q1XR8",
                   "Q1YL1", "Q1YR1", "Q1YL2", "Q1YR2", "Q1YL5", "Q1YR5", "Q1YL8", "Q1YR8"]

    quad_list =["QB3L1", "QA3L1", "QB2L1", "QA2L1", "QB1L1", "QA1L1",
                "QA1R1", "QB1R1", "QA2R1", "QB2R1", "QA3R1", "QB3R1"]

    plt.figure(figsize=(10,7))
    plt.plot(dbetas_0[0], 'o', label=r'before correction, $\langle\Delta\beta/\beta\rangle$ = %1.2f %%' % beating_0)
    plt.plot(dbetas[0], 'o', label=r'After correction, $\langle\Delta\beta/\beta\rangle$ = %1.2f %%' % beating)
    #plt.plot(betas_B1, label='nominal')
    plt.legend(fontsize=18)
    #plt.xlabel("Magnet number", fontsize=18)
    plt.ylabel(r"$\Delta\beta/\beta$ [%]", fontsize=18)
    plt.xticks(np.arange(0,16), magnet_list, rotation=70)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=18)
    plt.savefig("beating_after_crrection.png", bbox_to_anchor="tight")

    plt.figure(figsize=(10,7))
    plt.plot(errors/nominal, 'o', label='Errors')
    plt.plot(action_vec/nominal, 'o', label='Actions')
    #plt.plot(betas_B1, label='nominal')
    plt.legend(fontsize=18)
    plt.xlabel("Magnet number", fontsize=18)
    plt.ylabel(r"Relative magnet strength", fontsize=18)
    plt.xticks(np.arange(0,12), quad_list, rotation=70)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=18)
    plt.savefig("errors_and_actions_after_correction.png", bbox_to_anchor="tight")

    plt.figure(figsize=(10,7))
    plt.plot(1 - (nominal + errors)/nominal, 'o', label='State before correction')
    plt.plot(1- obs/nominal, 'o', label='State after correction')
    plt.legend(fontsize=18)
    plt.xlabel("Magnet number", fontsize=18)
    plt.ylabel(r"Relative magnet strength", fontsize=18)
    plt.xticks(np.arange(0,12), quad_list, rotation=70)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=18)
    plt.savefig("state_before_and_after_correction.png", bbox_to_anchor="tight")

    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    t0 = time()
    run_correction()
    print("Time required = %1.2f seconds" % float(time() - t0))