1. Open command prompt
2. Go to Custom-gym folder, where setup.py file is located and run pip install -e . command
3. Then run the following lines:
	import gym
	import envs
	
	env = gym.make('CustomEnv-v0') --> id of the environment used for registering, present in envs/__init__.py file
	env.start() ---> to open sumo gui with loaded configuration.