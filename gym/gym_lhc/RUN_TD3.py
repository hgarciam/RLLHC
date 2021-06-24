import envs.lhc_env as myenv
import numpy as np
import os
import matplotlib.pyplot as plt
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.td3.policies import MlpPolicy
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.monitor import Monitor

# TODO
# Normalize action/observation space [-1,1]
# Define reward in an better way?
# Tune hyperparameters


def run_rl(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'{name}')  # Press Ctrl+F8 to toggle the breakpoint.

    log_dir = "tmp/"
    os.makedirs(log_dir, exist_ok=True)
    env = myenv.LHCEnv()
    env = Monitor(env, log_dir)
    obs = env.reset()

    timesteps = 400

    n_actions = env.action_space.shape[-1]
    print("Number of actions in environment =", n_actions)
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # Apply TD3 policy
    print("TD3 Running...")
    model = TD3(MlpPolicy, env, action_noise=action_noise, verbose=1)
    print("TD3 Model Created. Now learning...")
    model.learn(total_timesteps=timesteps)
    print("TD3 learning done!")

    model.save("td3_LHC")

    obs = env.reset()
    plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "TD3 IR1 Optics Correction",[10, 5])

    plt.show()

    exit()

    done = False
    step = 0

    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        if done:
            beating = (-rewards+5)
            #print("Correction completed!")
            print("beta-beating after correction = %1.2f %%" % beating)
            #print("Correction =", action)
            #print("Final state =", obs)


if __name__ == '__main__':
    run_rl('Testing OpenAI Gym')
