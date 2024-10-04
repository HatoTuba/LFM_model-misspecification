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
from tensorflow.keras.backend import clear_session

############################################
from priors import ddm_prior, levy_prior
from likelihoods import _ddm_trial, ddm_process, levy_process
from context_gen import generate_condition_matrix
from models import Diffusion, Levy
from get_setup import get_setup


def train_network_offline(models):
    """This function runs trains the networks with the models we choose."""
    
    # gpu setting and checking
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
   
    data_paths = {
        "standard_ddm": ("data/train/standard_ddm_train_sim.pkl", "data/train/standard_ddm_val_sim.pkl"),
        "full_ddm": ("data/train/full_ddm_train_sim.pkl", "data/train/full_ddm_val_sim.pkl"),
        "standard_levy": ("data/train/standard_levy_train_sim.pkl", "data/train/standard_levy_val_sim.pkl"),
        "full_levy": ("data/train/full_levy_train_sim.pkl", "data/train/full_levy_val_sim.pkl"),
    }

    for model_name in models:
        
        simulations_dict_path, validation_sims_path = data_paths[model_name]

        with open(simulations_dict_path, 'rb') as f:
            simulations_dict = pickle.load(f)
        with open(validation_sims_path, 'rb') as f:
            validation_sims = pickle.load(f)

        exp = get_setup(model_name=model_name)
        history = exp.run(simulations_dict=simulations_dict, validation_sims=validation_sims, epochs=100, batch_size=32)

        # Save history plots with a unique name
        plot_name = f"plots/{model_name}_loss.png"
        f = bf.diagnostics.plot_losses(history['train_losses'], history['val_losses'])
        f.savefig(plot_name)
        print(f"Loss plot saved: {plot_name}")

        clear_session()

        

def train_network_online(models):
    # gpu setting and checking
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
    
    for model_name in models:
        exp = get_setup(model_name=model_name)
        history = exp.run(epochs=50, iterations_per_epoch=1000, batch_size=32)
        
        plot_name = f"plots/{model_name}_online_loss.png"
        f = bf.diagnostics.plot_losses(history)
        print(f"Loss plot saved: {plot_name}")

        clear_session()
        
    
    