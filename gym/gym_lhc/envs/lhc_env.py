import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
import matplotlib.pyplot as plt
import pickle
import random
import pandas as pd

class LHCEnv(gym.Env):
    """
    Description:
        The correction of the optics in the LHC is essential to ensure the luminosity
        performance of the collider. In this class, the objective is to create the environment
        where the agent takes actions based on a TD3 policy.

    Source:

    Observation:
        Type: Box(3)
        Num     Observation               Min                     Max
        0       Global betabeat IRs        -Inf                    Inf

    Actions:
        Type:Box()
        Num     Action
        0       corr_Q1
        1       corr_Q2
        2       corr_Q3

        Note: The action space is composed by the powering of the different corrector magnets.
        It is a continuous variable with range [-inf, +inf].


    Reward:

    Starting state:

    Episode Termination:
    """

    metadata = {'render.modes': ['human']}

    def __init__(self):

        def generate_triplet_errors(index,magnet):

            if magnet == 'Q2':
                Nq = 4     # Number of individual Q2, 2 ips, 2 quads per ip side
                FullRangeT = 50  # Full range of integrated gradient error in units
                measerror = 2  # Random error of measurement
                systerror = 10 # Random systematic error
                sorting = True

                magnet_list_L = ["MQXFB.A2L1", "MQXFB.B2L1"]
                magnet_list_R = ["MQXFB.A2R1", "MQXFB.B2R1"]

            elif magnet == 'Q1' or 'Q3':
                Nq = 4
                FullRangeT = 50
                measerror = 2
                systerror = 10
                sorting = False

                if magnet == 'Q1':
                    magnet_list_L = ["MQXFA.A1L1","MQXFA.B1L1"]
                    magnet_list_R = ["MQXFA.A1R1","MQXFA.B1R1"]
                else:
                    magnet_list_L = ["MQXFA.A3L1","MQXFA.B3L1"]
                    magnet_list_R = ["MQXFA.A3R1","MQXFA.B3R1"]

            magnet_list = magnet_list_L + magnet_list_R

            '''Random errors generation'''
            g, m, s = [], [], []
            gmsum = []
            gmssum = []
            pairpower = []

            stmp = systerror*2*(random.random()-0.5) # systematic error of measurement
            for i in range(Nq):
                gtmp = FullRangeT*(random.random()-0.5)
                mtmp = measerror*2*(random.random()-0.5)
                g.append(gtmp) # systematic error of measurement
                m.append(mtmp) # random measurement error
                s.append(stmp)
                gmsum.append(g+m)

            gm = list(zip(g,m))
            if sorting:
                gm.sort(key=sum)
            gmsum = [sum(x) for x in gm]
            gmssum = [sum(x) for x in zip(gmsum,s)]

            g = [row[0] for row in gm]
            m = [row[1] for row in gm]

            for i in range(0,len(gm)-1,2):
                pairpower.append((gmsum[i]+gmsum[i+1])/2)
                pairpower.append((gmsum[i]+gmsum[i+1])/2)

            gdiff = [-x + y for x, y in zip(g, pairpower)]
            gsdiff = [x + y for x, y in zip(gdiff, s)]

            data = {'Gradient error': g,
                    'Measurement error': m,
                    'Systematic error': s,
                    'Believed error': gmsum,
                    'Pair powered': pairpower,
                    'Difference': gdiff,
                    'Final error': gsdiff
            }

            df = pd.DataFrame(data,index=magnet_list)

            return df

        self.num_correctors = 6 # number of correctors used.
        self.min_action = -1.0
        self.max_action = 1.0

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

        self.betas_B1 = [betaX_L1, betaX_R1, betaX_L2, betaX_R2,
         betaX_L5, betaX_R5, betaX_L8, betaX_R8,
         betaY_L1, betaY_R1, betaY_L2, betaY_R2,
         betaY_L5, betaY_R5, betaY_L8, betaY_R8,
         ]

        Q1_error_df = generate_triplet_errors(1, 'Q1')
        Q2_error_df = generate_triplet_errors(1, 'Q2')
        Q3_error_df = generate_triplet_errors(1, 'Q3')

        QA1L1_error = Q1_error_df.at['MQXFA.A1L1', 'Final error']
        QB1L1_error = Q1_error_df.at['MQXFA.B1L1', 'Final error']
        QA1R1_error = Q1_error_df.at['MQXFA.A1R1', 'Final error']
        QB1R1_error = Q1_error_df.at['MQXFA.B1R1', 'Final error']

        QA2L1_error = Q2_error_df.at['MQXFB.A2L1', 'Final error']
        QB2L1_error = Q2_error_df.at['MQXFB.B2L1', 'Final error']
        QA2R1_error = Q2_error_df.at['MQXFB.A2R1', 'Final error']
        QB2R1_error = Q2_error_df.at['MQXFB.B2R1', 'Final error']

        QA3L1_error = Q3_error_df.at['MQXFA.A3L1', 'Final error']
        QB3L1_error = Q3_error_df.at['MQXFA.B3L1', 'Final error']
        QA3R1_error = Q3_error_df.at['MQXFA.A3R1', 'Final error']
        QB3R1_error = Q3_error_df.at['MQXFA.B3R1', 'Final error']

        error = np.array([QB3L1_error, QA3L1_error, QB2L1_error, QA2L1_error, QB1L1_error, QA1L1_error,
                 QA1R1_error, QB1R1_error, QA2R1_error, QB2R1_error, QA3R1_error, QB3R1_error])

        modelfile = 'Ridge_surrogate_20k.pkl'
        estimator = pickle.load(open(modelfile, "rb"))

        KQX3B1 = -0.0234346323
        KQX3B2 = 0.0234346323
        KQX2B1 = 0.04005580581
        KQX2B2 = -0.04005580581
        KQX1B1 = -0.02385208
        KQX1B2 = 0.02385208

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

        KQX = [KQX3B1, KQX3B1, KQX2B2, KQX2B2,
              KQX1B1, KQX1B1, KQX1B2, KQX1B2,
              KQX2B1, KQX2B1, KQX3B2, KQX3B2]

        #self.state = KQX + error*1e-4
        self.state = KQX
        print(self.state)

        modelfile = 'Ridge_surrogate_20k.pkl'
        self.estimator = pickle.load(open(modelfile, "rb"))

        self.num_correctors = 6
        self.num_magnets = 12

        self.action_space = spaces.Box(
            low=-1e-3,
            high=1e-3,
            shape=(self.num_correctors,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=(self.num_magnets,),
            dtype=np.float32
        )

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        KQX3B1, KQX3B1, KQX2B1, KQX2B1, KQX1B1, KQX1B1, KQX1B2, KQX1B2, KQX2B2, KQX2B2, KQX3B2, KQX3B2, = self.state

        KQX3B1 += action[0]
        KQX2B1 += action[1]
        KQX1B1 += action[2]
        KQX1B2 += action[3]
        KQX2B2 += action[4]
        KQX3B2 += action[5]

        input = KQX3B1, KQX3B1, KQX2B1, KQX2B1, KQX1B1, KQX1B1, KQX1B2, KQX1B2, KQX2B2, KQX2B2, KQX3B2, KQX3B2

        dbetas = self.estimator.predict(np.reshape(input, (1, -1)))/self.betas_B1*100
        beating = abs(dbetas).mean()

        done = bool(beating < 800)

        reward = 0
        if done:
            reward = 100.0
        reward -= beating

        self.state = np.array([ KQX3B1, KQX3B1, KQX2B1, KQX2B1, KQX1B1, KQX1B1, KQX1B2, KQX1B2, KQX2B2, KQX2B2, KQX3B2, KQX3B2])
        return self.state, reward, done, {}

    def reset(self):
        #self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
        return np.array(self.state)

  #def render(self, mode='human'): #
  #  ...

    def close(self):
        pass