import os, sys
sys.path.append("../")
sys.path.append("../DDM_LF_simstudy/simulators") 

import pickle
from tensorflow.keras.backend import clear_session

import numpy as np
from models import Diffusion, Levy
from experiment import DiffusionNet
from get_setup import get_setup
from tqdm import tqdm

# models = ("standard_ddm", "full_ddm", "standard_levy", "full_levy")
# sim_models = ('standard_ddm', 'standard_levy')        

        
def fit_models(models):        
    """This function fits the neural models, should be run standard and full models separately."""
    for model_name in tqdm(models):
        clear_session()
        exp = get_setup(model_name=model_name)
        for i, model_data in enumerate(models):
            sim_data_path = f"data/sim/{model_data}_sim.pkl"
            with open(sim_data_path, 'rb') as f:
                sim_data = pickle.load(f)
            configured_sim_data = exp.model.configure(sim_data)
            posterior_samples = exp.amortizer.sample(configured_sim_data, n_samples=1000)
            np.save(f'data/posteriors/posterior_{model_name}_{model_data}.npy', posterior_samples)

            
            
def fit_model_splitted_data(models):
    """This function fits large datasets by splitting them in 10 pieces."""
    for model_name in tqdm(models):
        clear_session()
        exp = get_setup(model_name=model_name)
        for i, model_data in enumerate(models):
            sim_data_path = f"data/sim/{model_data}_sim.pkl"
            with open(sim_data_path, 'rb') as f:
                sim_data = pickle.load(f)

            for i in range(10): # dictionary was splitted to 10
                parameters, summary_conditions = configure_splitting(sim_data)
                out_dict = dict(
                    parameters=parameters[i], 
                    summary_conditions=summary_conditions[i],
                )

                samples = exp.amortizer.sample(out_dict, n_samples=10, to_numpy=True)
                if i == 0:
                    posterior_samples = samples
                else:
                    posterior_samples = np.concatenate((posterior_samples, samples))
            np.save(f'data/posteriors/{model_name}_{model_data}.npy', posterior_samples)