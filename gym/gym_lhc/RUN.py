import envs.lhc_env as myenv
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

def run_rl(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'{name}')  # Press Ctrl+F8 to toggle the breakpoint.

    env = myenv.LHCEnv()
    obs = env.reset()

    for episode in range(1, 100):
        print('Episode = %i' % episode)
        # The noise objects for TD3
        n_actions = env.action_space.shape[-1]
        #action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        print(n_actions)

        #model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1)
        model = TD3("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=10000, log_interval=10)

        obs = env.reset()

    while True:
        print(action)
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        #env.render()



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run_rl('Testing OpenAI Gym')
