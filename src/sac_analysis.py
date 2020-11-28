import pickle
import matplotlib.pyplot as plt


rewards = pickle.load(open("soft_actor_critic/sac_reward.pkl", 'rb'))

plt.plot(range(1, len(rewards)+1), rewards)
plt.show()

