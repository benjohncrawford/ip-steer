import sys
from ._base_steer import Steer

# Vector-based Steers
from ._repe import RepE
from ._caa import CAA, MultiStepCAA
from ._iti import ITI

# OT-based Steers
from ._mimic import MiMiC
from ._lin_act import LinAcT

# ODESteer
from ._ode_steer import BaseODESteer, ODESteer, RFFODESteer
from ._step_ode_steer import BaseStepODESteer, StepODESteer, RFFStepODESteer  
from ._interior_point_steer import BaseIPSteer, IPSteer  

__all__ = [
    'Steer', 'VecSteer', 
    # Baselines
    'RepE', 'CAA', 'ITI', 'MiMiC', 'LinAcT', 'MultiStepCAA'
    # ODESteer
    'BaseODESteer', 'ODESteer', 'RFFODESteer',
    'BaseStepODESteer', 'StepODESteer', 'RFFStepODESteer', 'BaseIPSteer', 'IPSteer'
]

def get_steer_model(name: str, *args, **kwargs) -> type[Steer]:
    if name == "NoSteer":
        return None
    return getattr(sys.modules[__name__], name)(*args, **kwargs)