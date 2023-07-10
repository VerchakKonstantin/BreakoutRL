from env import init_gym_env
from agent import DQAgent

# Specify environment location
env_name = 'ALE/Breakout-v5'

# Initialize Gym Environment
env, state_space, action_space = init_gym_env(env_name)

# Create an agent
agent = DQAgent(replace_target_cnt=5000, env=env, state_space=state_space, action_space=action_space, model_name='breakout_model', gamma=.99,
                eps_strt=.1, eps_end=.001, eps_dec=5e-6, batch_size=32, lr=.001)

# Train num_eps amount of times and save onnx model
agent.train(num_eps=75000)