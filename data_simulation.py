import os, sys
sys.path.append("../")
sys.path.append("../DDM_LF_simstudy/simulators") 

import numpy as np
from functools import partial
import sys, os, scipy
import seaborn as sns
import tensorflow as tf
import bayesflow as bf
import pickle, Cython

from priors import *
from likelihoods import *
from context_gen import generate_condition_matrix
from get_setup import get_setup
from models import Diffusion, Levy

def generate_training_data(models, num_train, num_val):
    """
    This function simulates training and validation data for both of the models for later offline training.

    Batch size of training data is 100000 and validation data is 50.
    """    
    for model in models:
        exp = get_setup(model)
        train_sim = exp.model.generate(num_train)
        val_sim = exp.model.generate(num_val)
        
        with open(f'data/train/{model}_train_sim.pkl', 'wb') as f:
            pickle.dump(train_sim, f)
        with open(f'data/train/{model}_val_sim.pkl', 'wb') as f:
            pickle.dump(val_sim, f)


#####################                
                
def simulate_data_fix_alpha(simulate_data, num_sim):              
    """This function simulates data with fixed alpha values for later model fitting."""
    standard_ddm = Diffusion()
    full_ddm = Diffusion(standard=False)
    standard_levy_fix = Levy(fix_alpha=1.5)           
    full_levy_fix = Levy(standard=False, fix_alpha=1.5)
    standard_cauchy = Levy(fix_alpha=1.0)
    full_cauchy = Levy(standard=False, fix_alpha=1.0)

    standard_ddm_sim = standard_ddm.generate(num_sim)
    full_ddm_sim = full_ddm.generate(num_sim)
    standard_levy_fix_sim = standard_levy_fix.generate(num_sim)
    full_levy_fix_sim = full_levy_fix.generate(num_sim)
    standard_cauchy_fix_sim = standard_cauchy.generate(num_sim)
    full_cauchy_fix_sim = full_cauchy.generate(num_sim)

    simulation_data = [
        'standard_ddm_fix_sim',
        'full_ddm_fix_sim',
        'standard_levy_fix_sim',
        'full_levy_fix_sim',
        'standard_cauchy_fix_sim',
        'full_cauchy_fix_sim'
    ]

    if simulate_data:
        for data_name in simulation_data:
            with open(f'data/sim/sim_fix/{data_name}.pkl', 'wb') as f:
                pickle.dump(locals()[data_name], f)
    else: # load previously simulated data
        for data_name in simulation_data:
            with open(f'data/sim/sim_fix/{data_name}.pkl', 'rb') as f:
                locals()[data_name] = pickle.load(f)

            
############################

def simulate_data(models, num_sim):
    for model in models:
        exp = get_setup(model)
        sim = exp.model.generate(num_sim)

        with open(f'data/sim/{model}_sim.pkl', 'wb') as f:
            pickle.dump(sim, f)
