from gym.envs.registration import register

register(id='grSimSSLPenalty-v0',
    entry_point='gym_ssl.grsim_ssl:GrSimSSLPenaltyEnv'
    )

register(id='grSimSSLShootGoalie-v0',
    entry_point='gym_ssl.grsim_ssl:shootGoalieEnv'
    )

register(id='grSimSSLGoToBall-v0',
    entry_point='gym_ssl.grsim_ssl:goToBallEnv'
    )

register(id='grSimSSLGK-v0',
    entry_point='gym_ssl.grsim_ssl:goalieEnv'
    )
