from gym.envs.registration import register

register(id='GymSumo-v0',
    entry_point='envs.gym_sumo_dir:SumoEnv')