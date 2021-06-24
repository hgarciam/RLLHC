import numpy as np
import pickle
import envs.lhc_env as myenv

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

env = myenv.LHCEnv()
obs = env.reset()
input = obs

modelfile = 'Ridge_surrogate_20k.pkl'
estimator = pickle.load(open(modelfile, "rb"))
dbetas = estimator.predict(np.reshape(input, (1, -1))) / betas_B1 * 100
beating = abs(dbetas).mean()

print(beating)