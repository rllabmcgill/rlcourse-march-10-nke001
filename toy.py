import numpy.random as rng
import numpy as np
import matplotlib.pyplot as plt

states = ['a', 'b', 'c']

mu = {}

mu[('a','c')] = 0.2
mu[('a','b')] = 0.6
mu[('a','a')] = 0.2

mu[('b','a')] = 0.3
mu[('b','b')] = 0.3
mu[('b','c')] = 0.4

mu[('c','a')] = 0.2
mu[('c','b')] = 0.4
mu[('c','c')] = 0.4

pi = {}

pi[('a','c')] = 0.3
pi[('a','b')] = 0.6
pi[('a','a')] = 0.1

pi[('b','a')] = 0.4
pi[('b','b')] = 0.2
pi[('b','c')] = 0.4

pi[('c','a')] = 0.2
pi[('c','b')] = 0.6
pi[('c','c')] = 0.2

rewards = {}

rewards['a'] = 0.0
rewards['b'] = 1.0
rewards['c'] = -1.0

'''
    Pick the next state with a probability proportional to the policy
'''
def sample(policy,start_state):
    probs = []

    for state in states:
        probs.append(policy[(start_state,state)])

    assert sum(probs) == 1.0

    samp = rng.multinomial(n=1, pvals=probs).argmax()

    return states[samp]

def evaluate_policy(policy,start_state,num_steps):
    reward_sum = 0.0
    state = start_state

    for i in range(0, num_steps):
        
        state = sample(policy, state)
        reward_sum += rewards[state]

    return reward_sum

def importance_evaluate_policy(behavior_policy, target_policy, start_state, num_steps):
    reward_sum = 0.0
    state = start_state

    acc_imp = 1.0

    for i in range(0, num_steps):
        next_state = sample(behavior_policy, state)
        importance_weight = 1.0#target_policy[(state, next_state)] / behavior_policy[(state, next_state)]
        #print "imp", importance_weight, state, next_state
        acc_imp *= importance_weight
        reward_sum += rewards[next_state]
        state = next_state

    return reward_sum * acc_imp

def retrace_evaluate_policy(behavior_policy, target_policy, start_state, num_steps):
    reward_sum = 0.0
    state = start_state

    acc_imp = 1.0

    for i in range(0, num_steps):
        next_state = sample(behavior_policy, state)
        importance_weight = min(1.0, target_policy[(state, next_state)] / behavior_policy[(state, next_state)])
        #print "imp", importance_weight, state, next_state
        acc_imp *= importance_weight
        reward_sum += rewards[next_state]
        state = next_state

    return reward_sum * acc_imp


if __name__ == "__main__":

    num_steps = 10
    num_samples = 100000

    '''
        Evaluating with the actual target policy (just to get ground truth for evaluation)
    '''
    true_rewards = []
    for j in range(0,num_samples):
        true_rewards.append(evaluate_policy(pi, 'a', num_steps))

    true_rewards = np.array(true_rewards)

    print "true reward under target policy", true_rewards.mean()

    '''
        Evaluating the target policy with an importance sampling estimator
    '''
    samples_imp = []
    for j in range(0,num_samples):
        samples_imp.append(importance_evaluate_policy(mu, pi, 'a', num_steps))

    samples_imp = np.array(samples_imp)

    plt.hist(samples_imp)
    plt.show()

    print samples_imp.mean()
    print samples_imp.std()

    samples_imp = []
    for j in range(0,num_samples):
        samples_imp.append(retrace_evaluate_policy(mu, pi, 'a', num_steps))

    samples_imp = np.array(samples_imp)

    plt.hist(samples_imp,bins=500)
    plt.show()

    print samples_imp.mean()
    print samples_imp.std()



