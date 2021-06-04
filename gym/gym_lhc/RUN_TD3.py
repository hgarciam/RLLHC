import envs.lhc_env as myenv
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.td3.policies import MlpPolicy

# TODO
# Normalize action/observation space [-1,1]
# Define reward in an better way?
# Tune hyperparameters


def run_rl(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'{name}')  # Press Ctrl+F8 to toggle the breakpoint.

    env = myenv.LHCEnv()
    obs = env.reset()

    n_episodes = 1
    timesteps = 2

    for episode in range(1, n_episodes+1):
        print('Episode %i' % episode)
        # The noise objects for TD3
        n_actions = env.action_space.shape[-1]
        print("Number of actions in environment =", n_actions)
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

        # Apply TD3 policy
        print("TD3 Running...")
        model = TD3(MlpPolicy, env, action_noise=action_noise, seed=1234, verbose=1)
        print("TD3 Model Created. Now learning...")
        model.learn(total_timesteps=timesteps)
        print("TD3 learning done!")

        #model.save("td3_pendulum")
        #model = TD3.load("td3_pendulum")

        obs = env.reset()

    exit()

    dones = False
    step = 0

    while not dones:
    #for i in range(100):
        print("Step =", step)
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        print(rewards, dones)
        step += 1
        #env.render()        


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run_rl('Testing OpenAI Gym')
