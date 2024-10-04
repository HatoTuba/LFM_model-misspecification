# models
import numpy as np
from scipy import stats
from scipy.stats import levy_stable
import bayesflow as bf
from functools import partial

from priors import standard_ddm_prior, ddm_prior, standard_levy_prior, levy_prior
from simulators import likelihoods 
from likelihoods import standard_ddm_process, ddm_process, standard_levy_process, levy_process
from context_gen import generate_condition_matrix, random_n_obs

class Diffusion:
    """
    A wrapper for a Diffusion decision process
    
    standard=True is the standard model
    standard=False is the full model
    """

    def __init__(self, standard=True, n_obs=500): 

        if standard == True:
            self.prior = bf.simulation.Prior(
                prior_fun=standard_ddm_prior,
            )
        else:
            self.prior = bf.simulation.Prior(
                prior_fun=ddm_prior,
            )
        self.context_generator = bf.simulation.ContextGenerator(
            batchable_context_fun=partial(generate_condition_matrix, n_obs),
        )
        if standard == True:
            self.likelihood = bf.simulation.Simulator(
            simulator_fun=standard_ddm_process,
            context_generator=self.context_generator,
        )
        elif standard == False:
            self.likelihood = bf.simulation.Simulator(
                simulator_fun=ddm_process,
                context_generator=self.context_generator,
            )

        self.generator = bf.simulation.GenerativeModel(
            prior=self.prior,
            simulator=self.likelihood,
            name="diffusion_model",
        )

    def generate(self, batch_size):
        return self.generator(batch_size)
    
    def context(self):
        return self.context_generator()
    
    def configure(self, raw_dict):
        rt = raw_dict['sim_data'][:, :, :1].astype(np.float32)
        context = np.array(raw_dict['sim_batchable_context'])[:, :, np.newaxis]
        summary_conditions = np.c_[rt, context].astype(np.float32)
        
        parameters = raw_dict['prior_draws'].astype(np.float32)

        out_dict = dict(
            parameters=parameters, 
            summary_conditions=summary_conditions,
        )

        return out_dict


class Levy:
    """
    standard=True is the standard model
    standard=False is the full model
    """
   
    def __init__(self, standard=True, n_obs=500, fix_alpha=False):
        
        if standard==True:
            self.prior = bf.simulation.Prior(
            prior_fun=partial(standard_levy_prior, fix_alpha=fix_alpha),
        )
        else:
            self.prior = bf.simulation.Prior(
            prior_fun=partial(levy_prior, fix_alpha=fix_alpha),
        )

#         self.context_generator = bf.simulation.ContextGenerator( #levyde calismayacak gibi 
#             non_batchable_context_fun=partial(random_n_obs, min_obs, max_obs),
#             batchable_context_fun=generate_condition_matrix,
# #             batchable_context_fun=partial(generate_condition_matrix, n_obs),
#             use_non_batchable_for_batchable=True,
#         )
        self.context_generator = bf.simulation.ContextGenerator(
            batchable_context_fun=partial(generate_condition_matrix, n_obs),
        )
        
        
        if standard==True:
            self.likelihood = bf.simulation.Simulator(
                simulator_fun=standard_levy_process,
                context_generator=self.context_generator,
            )
        else:
             self.likelihood = bf.simulation.Simulator(
                simulator_fun=levy_process,
                context_generator=self.context_generator,
            )

        self.generator = bf.simulation.GenerativeModel(
            prior=self.prior,
            simulator=self.likelihood,
            name="lf_model",
        )

    def generate(self, batch_size):
        return self.generator(batch_size)
    
    def context(self):
        return self.context_generator()
    
    def configure(self, raw_dict):
        rt = raw_dict['sim_data'][:, :, :1].astype(np.float32)
        context = np.array(raw_dict['sim_batchable_context'])[:, :, np.newaxis]
        summary_conditions = np.c_[rt, context].astype(np.float32)
        
        parameters = raw_dict['prior_draws'].astype(np.float32)

        out_dict = dict(
            parameters=parameters, 
            summary_conditions=summary_conditions,
        )

        return out_dict