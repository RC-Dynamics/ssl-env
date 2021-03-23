from gym.envs.registration import register

register(id='grSimSSLPenalty-v0',
    entry_point='gym_ssl.grsim_ssl:GrSimSSLPenaltyEnv'
    )

register(id='grSimSSLShootGoalie-v01',
    entry_point='gym_ssl.grsim_ssl:shootGoalieEnv',
    kwargs={'sparce_reward': True, "move_goalie": True}
    )

register(id='grSimSSLShootGoalie-v11',
    entry_point='gym_ssl.grsim_ssl:shootGoalieEnv',
    kwargs={'sparce_reward': False, "move_goalie": True}
    )

register(id='grSimSSLShootGoalie-v00',
    entry_point='gym_ssl.grsim_ssl:shootGoalieEnv',
    kwargs={'sparce_reward': True, "move_goalie": False}
    )

register(id='grSimSSLShootGoalie-v10',
    entry_point='gym_ssl.grsim_ssl:shootGoalieEnv',
    kwargs={'sparce_reward': False, "move_goalie": False}
    )

register(id='grSimSSLPass-v0',
    entry_point='gym_ssl.grsim_ssl:passEnv',
    kwargs={}
    )


register(id='grSimSSLGoToBall-v0',
    entry_point='gym_ssl.grsim_ssl:goToBallEnv'
    )

register(id='grSimSSLGK-v0',
    entry_point='gym_ssl.grsim_ssl:goalieEnv'
    )
