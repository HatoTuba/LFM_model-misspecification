import sys
sys.path.append("../DDM_LF_simstudy/simulators") 

from experiment import DiffusionNet, LevyNet
from models import Diffusion, Levy 

def get_setup(model_name):
    if model_name == "standard_ddm":
        exp = DiffusionNet(
            standard=True,
            checkpoint_path = f"checkpoints/ddm/standard_ddm"
        )
    if model_name == "full_ddm":
        exp = DiffusionNet(
            standard=False,
            checkpoint_path = f"checkpoints/ddm/full_ddm"
        )
    if model_name == "standard_levy":
        exp = LevyNet(
            standard=True,
            checkpoint_path = f"checkpoints/levy/standard_levy")
    if model_name == "full_levy":
        exp = LevyNet(
            standard=False,
            checkpoint_path = f"checkpoints/levy/full_levy")
    
    return exp