import numpy as np
import matplotlib.pyplot as plt

num_arms = 10
num_users = 2000
num_iter = 1000
epsilon = [0, 0.01, 0.1, 0.2]
np.random.seed(0)

arm_rewards = np.random.normal(size=num_arms)

for ep in epsilon:
    greedy_prob = 1 - ep + (ep/num_arms)
    exploration_prob = ep/num_arms

    user_rewards = {}
    average_reward = []

    # Initialize Reward for all users to 0 => Initial Bias
    for i in range(num_users):
        user_rewards[i] = {'total_avg_reward': 0,
                           'curr_opt_arm': np.random.choice(num_arms),
                           'num_arm_chosen': [0]*num_arms, # Number of times that arm has been chosen
                           'avg_reward_per_arm': [0]*num_arms}

    # Select an action for every user for every iteration
    for trial in range(1, num_iter+1):

        avg_reward_per_trial = 0
        for user in range(num_users):

            prob = [exploration_prob]*num_arms
            exploitation = user_rewards[user]['curr_opt_arm']
            prob[exploitation] = greedy_prob

            # Choose an arm
            choice = np.random.choice(range(num_arms), p=prob)

            # Choose a reward for that arm
            reward = arm_rewards[choice] + np.random.normal()

            # Update Total Reward, Average Reward for Chosen Arm
            user_rewards[user]['total_avg_reward'] = user_rewards[user]['total_avg_reward'] \
                                                     + (1/trial)*(reward - user_rewards[user]['total_avg_reward'])

            user_rewards[user]['num_arm_chosen'][choice] += 1

            user_rewards[user]['avg_reward_per_arm'][choice] = user_rewards[user]['avg_reward_per_arm'][choice] \
                                                       + (1/user_rewards[user]['num_arm_chosen'][choice])\
                                                       * (reward - user_rewards[user]['avg_reward_per_arm'][choice])

            # Update Optimal Arm after new reward, exploitation is the current optimal arm
            if user_rewards[user]['avg_reward_per_arm'][choice] > user_rewards[user]['avg_reward_per_arm'][exploitation]:
                user_rewards[user]['curr_opt_arm'] = choice

            avg_reward_per_trial += user_rewards[user]['total_avg_reward']

        avg_reward_per_trial /= num_users
        average_reward.append(avg_reward_per_trial)

    plt.plot(range(num_iter), average_reward, label='ε = ' + str(ep))

plt.xlabel('Number of Iterations')
plt.ylabel('Average Sum Reward')
plt.legend()
plt.show()
