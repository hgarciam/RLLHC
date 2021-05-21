import envs.lhc_env as myenv
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

# TODO
# Normalize action/observation space [-1,1]
# Define reward in an better way?
# Tune hyperparameters


def run_rl(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'{name}')  # Press Ctrl+F8 to toggle the breakpoint.

    env = myenv.LHCEnv()
    obs = env.reset()

    n_episodes = 200
    timesteps = 2

    for episode in range(1, n_episodes+1):
        print('Episode %i' % episode)
        # The noise objects for TD3
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

        print("TD3 Running...")
        model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1)
        print("TD3 Model Created. Now learning...")
        model.learn(total_timesteps=timesteps, log_interval=10)
        print("TD3 learning done!")

        obs = env.reset()

    while True:
        action, _states = model.predict(obs)
        print(action)
        obs, rewards, dones, info = env.step(action)
        print(rewards, dones)
        #env.render()



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run_rl('Testing OpenAI Gym')
