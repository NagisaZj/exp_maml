from gym.envs.registration import register

# Bandit
# ----------------------------------------

for k in [5, 10, 50]:
    register(
        'Bandit-K{0}-v0'.format(k),
        entry_point='maml_rl.envs.bandit:BernoulliBanditEnv',
        kwargs={'k': k}
    )

# TabularMDP
# ----------------------------------------

register(
    'TabularMDP-v0',
    entry_point='maml_rl.envs.mdp:TabularMDPEnv',
    kwargs={'num_states': 10, 'num_actions': 5},
    max_episode_steps=10
)

# Mujoco
# ----------------------------------------

register(
    'AntVel-v1',
    entry_point='maml_rl.envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'maml_rl.envs.mujoco.ant:AntVelEnv'},
    max_episode_steps=200
)

register(
    'AntDir-v1',
    entry_point='maml_rl.envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'maml_rl.envs.mujoco.ant:AntDirEnv'},
    max_episode_steps=200
)

register(
    'AntPos-v0',
    entry_point='maml_rl.envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'maml_rl.envs.mujoco.ant:AntPosEnv'},
    max_episode_steps=200
)

register(
    'HalfCheetahVel-v1',
    entry_point='maml_rl.envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'maml_rl.envs.mujoco.half_cheetah:HalfCheetahVelEnv'},
    max_episode_steps=200
)

register(
    'HalfCheetahDir-v1',
    entry_point='maml_rl.envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'maml_rl.envs.mujoco.half_cheetah:HalfCheetahDirEnv'},
    max_episode_steps=200
)

register(
    'ant-goal-sparse-v0',
    entry_point='maml_rl.envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'maml_rl.envs.mujoco_envs.ant_goal:AntGoalEnvSparse'},
    max_episode_steps=64
)


# 2D Navigation
# ----------------------------------------

register(
    '2DNavigation-v0',
    entry_point='maml_rl.envs.navigation:Navigation2DEnv',
    max_episode_steps=100
)

register(
    '2DPointEnvCorner-v0',
    entry_point='maml_rl.envs.point_envs.point_env_2d_corner:MetaPointEnvCorner',
    max_episode_steps=100

)

register(
    '2DPointEnvCorner-v1',
    entry_point='maml_rl.envs.point_envs.point_env_2d_momentum:MetaPointEnvMomentum',
    max_episode_steps=100
)

register(
    '2DPointEnvCustom-v1',
    entry_point='maml_rl.envs.point_envs.point_env_2d_custom:MetaPointEnvCustom',
    max_episode_steps=32
)

# Mujoco Envs
# ---------------------------------------

register(
    'AntRandDirecEnv-v1',
    entry_point='maml_rl.envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'maml_rl.envs.mujoco_envs.ant_rand_direc:AntRandDirecEnv', 'promp_env': True},
    max_episode_steps=100)
register(
    'AntRandDirec2DEnv-v1',
    entry_point='maml_rl.envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'maml_rl.envs.mujoco_envs.ant_rand_direc_2d:AntRandDirec2DEnv', 'promp_env': True},
    max_episode_steps=100)
register(
    'AntRandGoalEnv-v1',
    entry_point='maml_rl.envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'maml_rl.envs.mujoco_envs.ant_rand_goal:AntRandGoalEnv', 'promp_env': True},
    max_episode_steps=64)


register(
    'HalfCheetahRandDirecEnv-v1',
    entry_point='maml_rl.envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'maml_rl.envs.mujoco_envs.half_cheetah_rand_direc:HalfCheetahRandDirecEnv', 'promp_env': True},
    max_episode_steps=100)
register(
    'HalfCheetahRandVelEnv-v1',
    entry_point='maml_rl.envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'maml_rl.envs.mujoco_envs.half_cheetah_rand_vel:HalfCheetahRandVelEnv', 'promp_env': True},
    max_episode_steps=200)


register(
    'HumanoidRandDirecEnv-v1',
    entry_point='maml_rl.envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'maml_rl.envs.mujoco_envs.humanoid_rand_direc:HumanoidRandDirecEnv', 'promp_env': True, 'humanoid': True},
    max_episode_steps=200)
register(
    'HumanoidRandDirec2DEnv-v1',
    entry_point='maml_rl.envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'maml_rl.envs.mujoco_envs.humanoid_rand_direc_2d:HumanoidRandDirec2DEnv', 'promp_env': True, 'humanoid':True},
    max_episode_steps=200)


register(
    'Walker2DRandDirecEnv-v1',
    entry_point='maml_rl.envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'maml_rl.envs.mujoco_envs.walker2d_rand_direc:Walker2DRandDirecEnv', 'promp_env': True},
    max_episode_steps=200)
register(
    'Walker2DRandVelEnv-v1',
    entry_point='maml_rl.envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'maml_rl.envs.mujoco_envs.walker2d_rand_vel:Walker2DRandVelEnv', 'promp_env': True},
    max_episode_steps=200)


register(
    'Walker2DRandParamsEnv-v1',
    entry_point='maml_rl.envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'rand_param_envs.walker2d_rand_params:Walker2DRandParamsEnv', 'promp_env': True},
    max_episode_steps=64)
register(
    'HopperRandParamsEnv-v1',
    entry_point='maml_rl.envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'rand_param_envs.hopper_rand_params:HopperRandParamsEnv', 'promp_env': True},
    max_episode_steps=64)
register(
    'reacher-goal-sparse-v1',
    entry_point='maml_rl.envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'maml_rl.envs.mujoco_envs.reacher_goal:ReacherGoalEnv_sparse', 'promp_env': True},
    max_episode_steps=32)