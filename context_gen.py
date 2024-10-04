import numpy as np 

# non batchable context
def random_n_obs(min_obs, max_obs):
    return np.random.randint(low=min_obs, high=max_obs + 1)


# batchable context
def generate_condition_matrix(n_obs, n_conditions=2):
    obs_per_condition = np.ceil(n_obs/n_conditions)
    condition = np.arange(n_conditions)
    condition = np.repeat(condition, obs_per_condition)
    return condition[:n_obs]


# # batchable_context
# def generate_condition_matrix(n_obs, n_conditions=2):
#     obs_per_condition = np.ceil(n_obs / n_conditions)
#     condition = np.arange(n_conditions)
#     condition = np.repeat(condition, obs_per_condition)
#     np.random.shuffle(condition)
#     out = np.array(condition[:n_obs], dtype=np.int32)
#     return out