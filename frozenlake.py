"""

"""




import gym
import numpy as np

import os
import time

env = gym.make('FrozenLake-v0')
Q = np.random.rand(env.observation_space.n, env.action_space.n)

lr = 1e-3
dr = .5
epis = 5000

rewards = []

for i in range(epis):
    s = env.reset()
    epi_r = 0
    f = False
    j = 0
    while j < 99:
        os.system('clear')
        print('episode {}'.format(i))
        env.render()
        time.sleep(.5)
        j += 1
        a = np.argmax(Q[s,:])
        s1, r, f, _ = env.step(a)
        Q[s, a] = Q[s, a] + lr * (r + dr * np.max(Q[s1, :]) - Q[s, a])
        epi_r += r
        s = s1
        if f:
            break
    rewards.append(epi_r)

print(Q)
print(rewards)
