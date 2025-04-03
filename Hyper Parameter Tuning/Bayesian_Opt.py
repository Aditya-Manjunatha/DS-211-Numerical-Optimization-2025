# Derivative free optimization 
# 1) Bayesian Opt
# 2) Genetic Algorithms

import numpy as np
from sklearn.neighbors import KernelDensity

def f(x1, x2) :
    z = (x1-2)**2 + (x2-1)**2
    return z

x0 = np.array([0, 0])
x = x0
iter = 0
iter_max = 100
threshold = 0.01
stopping_criteria = False


# Genral derivative based step method to fidnd the optima
# while stopping_criteria == False:

#     step = derivative_based_step(f, x)

#     x = x + step 

#     if step < threshold:
#         stopping_criteria = True

#     iter = iter + 1

#     if iter == iter_max:
#         stopping_criteria = True

"""

# In bayesian optimization we do a proposal distribution instead of the step

How to get the proposal distribution?

We use a markov chain to sample the proposal distribution. Specifically the metropolis hastings algorithm.

Then we need to say if we want to accept the proposal or not.
"""
innovation_threshold = 0.2 # 20% if the time if will explore and 80% it will exploit
# So 20% of the time we will try to be more explorative and see if its better or not that the current x_new

proposal_accepted = False

def proposal_distribution(f, x_old, frac = 0.1):
    # We will use a normal distribution to sample the proposal distribution
    # We can also use a uniform distribution or any other distribution
    # But we will use a normal distribution for now

    # What is a good candidate for scale ?
    # We will move only a small amount from the current point for exploration

    x_new = np.random.normal(loc = x_old, scale =  frac * abs(x_old) + 1e-8, size = 1)
    return x_new


while stopping_criteria == False:
    while proposal_accepted == False:
        x_old = x
        f_now = f(x_old)
        x_new = proposal_distribution(f, x_old)
        f_new = f(x_new)
        # Metropolis hastings algorithm
        if f_new < f_now:
            # Means the new point has a lower functional value than x_old
            # But we will still not accept this x_new
            # Will accept only if we are in explaitation mode
            # If we are in exploration mode, we will do the following
            accept_rn = np.random.rand(1)

            if accept_rn > innovation_threshold:
                x = x_new
                proposal_accepted == True

            else:
                proposal_accepted == False

        else:
            proposal_accepted == False

    iter = iter + 1

    if iter == iter_max:
        stopping_criteria = True

x_soln = x 

"""
In the above code we only gave one proposal point
Now we will give multiple proposal points
"""

def new_proposal_distribution(f, x_old, frac = 0.1):
    x_samples = np.random.normal(loc = x_old, scale = 1.0, size = 100)
    f_samples = f(x_samples)

    # Finding f_samples is very expensive beacuse we have to evaluate the function at each point
    # Parzen estimation :-
    # We make 2 buckets
    goodness_threshold = ...
    good_samples = x_samples[f_samples < goodness_threshold]
    bad_samples = x_samples[f_samples >= goodness_threshold]

    good_kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(good_samples)
    bad_kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(bad_samples)

    new_samples = good_kde.KernelDensity.sample(100)
    ratio = good_kde.score_samples(new_samples) / bad_kde.score_samples(new_samples)

    final_accepted_sample = new_samples[np.argmax(ratio)]

    return final_accepted_sample

# What is the difference between a Gaussian mixture model and the KS density fit ?
# GMM is a semi paramteric estimate
# in Ksdensity, each datapoint gets a small gaussian on it

# 1:24:00 :- Formulating LP program from wordproblem :- Very important
# IMportant things for quiz :- 1:27:00