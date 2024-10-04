from abc import ABC, abstractmethod
import bayesflow as bf
from models import Diffusion, Levy 

class Experiment(ABC):
    @abstractmethod
    def __init__(self, model):
        pass

    @abstractmethod
    def run(self):
        pass

class DiffusionNet(Experiment):  
    def __init__(self, standard, checkpoint_path): 
        self.summary_net = bf.summary_networks.SetTransformer(
            input_dim=2,
            summary_dim=20
            )
        if standard == True:
            self.model = Diffusion(standard=True)
            self.inference_net = bf.networks.InvertibleNetwork(
                num_params=5,  # put priors 
                num_coupling_layers=6,
                coupling_design='spline'
                # only for online training
#                 coupling_settings={"dense_args": dict(kernel_regularizer=None), "dropout": False}
                )
        else:
            self.model = Diffusion(standard=False)
            self.inference_net = bf.networks.InvertibleNetwork(
                num_params=8,  # put priors 
                num_coupling_layers=6,
                coupling_design='spline'
                # only for online training
#                 coupling_settings={"dense_args": dict(kernel_regularizer=None), "dropout": False}
                )
        self.amortizer = bf.amortizers.AmortizedPosterior(
            self.inference_net,
            self.summary_net,
            name='ddm_amortizer'
            )
        self.trainer = bf.trainers.Trainer(
            amortizer=self.amortizer,
            generative_model=self.model.generate,
            configurator=self.model.configure,
            checkpoint_path=checkpoint_path
        )
        
    def run(self, simulations_dict, validation_sims, epochs=75, batch_size=32):
        history = self.trainer.train_offline(simulations_dict=simulations_dict, 
                                             validation_sims=validation_sims, 
                                             epochs=epochs, batch_size=batch_size)
        return history
        
#     def run(self, epochs=75, iterations_per_epoch=1000, batch_size=32):
#         history = self.trainer.train_online(epochs, iterations_per_epoch, batch_size)
#         return history

########################################################################
class LevyNet(Experiment):    
    def __init__(self, standard, checkpoint_path): 
        self.summary_net = bf.summary_networks.SetTransformer(
            input_dim=2,
            summary_dim=20
            )
        if standard == True:
            self.model = Levy(standard=True)
            self.inference_net = bf.networks.InvertibleNetwork(
                num_params=6,  # put priors 
                num_coupling_layers=6,
                coupling_design='spline'
                # only for online training
#                 coupling_settings={"dense_args": dict(kernel_regularizer=None), "dropout": False}
            )
        else:
            self.model = Levy(standard=False)
            self.inference_net = bf.networks.InvertibleNetwork(
                num_params=9,  # put priors 
                num_coupling_layers=6,
                coupling_design='spline'
#                 coupling_settings={"dense_args": dict(kernel_regularizer=None), "dropout": False}
            )
        self.amortizer = bf.amortizers.AmortizedPosterior(
            self.inference_net,
            self.summary_net,
            name='lf_amortizer'
            )
        self.trainer = bf.trainers.Trainer(
            amortizer=self.amortizer,
            generative_model=self.model.generate,
            configurator=self.model.configure,
            checkpoint_path=checkpoint_path
        )

# To run offline training un-comment this part
    def run(self, simulations_dict, validation_sims, epochs=75, batch_size=32):
        history = self.trainer.train_offline(simulations_dict=simulations_dict, 
                                             validation_sims=validation_sims, 
                                             epochs=epochs, batch_size=batch_size)
        return history
        
        

#     def run(self, epochs=75, iterations_per_epoch=1000, batch_size=32):
#         history = self.trainer.train_online(epochs, iterations_per_epoch, batch_size)
#         return history


