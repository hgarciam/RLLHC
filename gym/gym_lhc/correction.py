import envs.lhc_env as myenv
from stable_baselines3 import TD3
import matplotlib.pyplot as plt
import pickle
import numpy as np
from time import time


def run_correction():

    KQX3B1 = -0.0234346323
    KQX2B1 = 0.04005580581
    KQX1B1 = -0.02385208
    KQX1B2 = 0.02385208
    KQX2B2 = -0.04005580581
    KQX3B2 = 0.0234346323

    input = [KQX3B1, KQX3B1, KQX2B1, KQX2B1,
             KQX1B1, KQX1B1, KQX1B2, KQX1B2,
             KQX2B2, KQX2B2, KQX3B2, KQX3B2]

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

    # Load environment
    env = myenv.LHCEnv()
    obs = env.reset()
    errors = obs/input-1
    print("State:", obs)
    print("Errors:", errors)
    initial = obs
    obs_0 = obs

    # Estimate initial beta-beating
    modelfile = 'Ridge_surrogate_20k.pkl'
    estimator = pickle.load(open(modelfile, "rb"))
    dbetas_0 = estimator.predict(np.reshape(initial, (1, -1))) / betas_B1 * 100
    beating_0 = abs(dbetas_0).mean()
    print("beta-beating before correction = %1.2f %%" % beating_0)

    # Load RL model
    model = TD3.load("td3_LHC_136")

    done = False
    step = 0

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

    obs_0_norm = (obs_0-input)/input*100
    obs_norm = (obs-input)/input*100

    '''
    plt.figure()
    plt.plot(obs_0_norm, 'o', label='Before correction')
    plt.plot(obs_norm, 'o', label='After Correction')
    plt.axhline(0, label='Perfect machine')
    plt.legend()
    plt.show()
    '''

    rel_action = (np.array([action[0], action[0], action[1], action[1], action[2], action[2],
                  action[3], action[3], action[4], action[4], action[5], action[5]]))/np.array(input)

    action_vec = (np.array([action[0], action[0], action[1], action[1], action[2], action[2],
                  action[3], action[3], action[4], action[4], action[5], action[5]]))

    #print("Relative action:", rel_action)
    print("Absolute action:", action_vec)

    magnet_list = ["Q1XL1", "Q1XR1", "Q1XL2", "Q1XR2", "Q1XL5", "Q1XR5", "Q1XL8", "Q1XR8",
                   "Q1YL1", "Q1YR1", "Q1YL2", "Q1YR2", "Q1YL5", "Q1YR5", "Q1YL8", "Q1YR8"]

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
    plt.savefig("beating_after_crrection_no_errors.png")

    plt.figure()
    plt.plot(errors, 'o', label='Errors')
    plt.plot(action_vec, 'o', label='Actions')
    #plt.plot(betas_B1, label='nominal')
    plt.legend(fontsize=18)
    plt.xlabel("Magnet number", fontsize=18)
    plt.ylabel(r"$\Delta\beta/\beta$ [%]", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    t0 = time()
    run_correction()
    print("Time required = %1.2f seconds" % float(time() - t0))