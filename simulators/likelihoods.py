import numpy as np
from scipy import stats
from scipy.stats import levy_stable
from numba import njit
from priors import *
# from levy import levy_trial

#################
# Setting up the cython

import ctypes
from numba.extending import get_cython_function_address
import levy

# Get a pointer to the C function levy.c
addr_levy= get_cython_function_address("levy", "levy_trial")
functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double,                            
                            ctypes.c_double, ctypes.c_double, ctypes.c_double,
                            ctypes.c_double, ctypes.c_int)
levy_trial = functype(addr_levy)
###########
# condition for min RT
min_rt = 0.03 
max_rt = 20

@njit
def _ddm_trial(v, a, bias, ndt, dt=0.001, s=1.0, max_iter=1e5): #used just internally
    """Generates a response time of a single trial from a diffusion decision process.
    
    Parameters:
    -----------
    v       : float
        The drift rate parameter.
    a       : float
        The boundary separation parameter.
    bias    : float
        The starting point parameter.
    ndt     : float
        The non-decision time parameter.
    dt      : float, optional, default: 0.001
        Time resolution of the process. 
        Default corresponds to a precision of 1 milisecond.
    s       : float, optional, default: 1
        Scaling factor of the Wiener noise.
    max_iter: int, optional, default: 1e5
        Maximum iteration of the process. 
        Default corresponds to 100 seconds.
    
    Returns:
    -----------
    rt      : float
        A response time samples from the Diffusion decision process.
        Reaching upper boundary results in positive, lower boundary results in negative rt's.    
    """

    n_iter = 0
    x = a * bias
    c = np.sqrt(dt *s)

    while x >= 0 and x <= a and n_iter < max_iter:
        x += v*dt + c * np.random.randn()
        n_iter += 1
    
    rt = n_iter * dt
    return rt+ndt if x >= 0 else -(rt+ndt)

@njit
def ddm_process(theta, context): 
    """Generates a single simulation from a full Diffusion decision process with random variability.

    Parameters:
    -----------
    theta       : np.ndarray of shape (8, )
        The 8 latent DDM parameters: v1, v2, v_s, a, bias, bias_s, ndt, ndt_s
    context     : np.ndarray of shape (2, n_obs)
        Consists of 0 and 1s indicating lower (0) and upper (1) boundaries.
    
    Returns:
    -----------
    rt : np.ndarray of shape (n_obs, 1)
        Response time samples from the Diffusion decision process.
        Reaching upper boundary results in positive, lower boundary results in negative rt's. 
    """

    n_obs   = context.shape[0] # n_obs is extracted from the context
    rt      = np.zeros(n_obs)
    v_t     = np.zeros(n_obs)
    
    v1, v2, v_s, a, bias, bias_s, ndt, ndt_s = theta
        
    # across-trial variability parameters for starting point and non-decision times
    bias_t = np.random.uniform(theta[4] - theta[5]/2, theta[4] + theta[5]/2, size=n_obs)
    ndt_t  = np.random.uniform(theta[6] - theta[7]/2, theta[6] + theta[7]/2, size=n_obs)
    
    for n in range(n_obs):
        v_t[n] = theta[int(context[n])] + theta[2] * np.random.randn() 
#         rt[n] = _ddm_trial(v_t[n], theta[3], bias_t[n], ndt_t[n])
        temp_rt = _ddm_trial(v_t[n], theta[3], bias_t[n], ndt_t[n])
#         while np.abs(temp_rt) < min_rt or np.abs(temp_rt) > max_rt:          # gets stuck   
#             temp_rt = _ddm_trial(v_t[n], theta[3], bias_t[n], ndt_t[n])
                        
        rt[n] = temp_rt

    return rt[:, np.newaxis]


@njit
def standard_ddm_process(theta, context):
    """Generates a single simulation from a standard Diffusion decision process with random variability.

    Parameters:
    -----------
    theta       : np.ndarray of shape (5, )
        The 5 latent DDM parameters: v1, v2, a, bias, ndt
    context     : np.ndarray of shape (2, n_obs)
        Consists of 0 and 1s indicating lower (0) and upper (1) boundaries.
    
    Returns:
    -----------
    rt : np.ndarray of shape (n_obs, 1)
        Response time samples from the Diffusion decision process.
        Reaching upper boundary results in positive, lower boundary results in negative rt's. 
    """
    # theta = v1, v2, a, bias, ndt
    n_obs   = context.shape[0] # n_obs is extracted from the context
    rt      = np.zeros(n_obs)
    v_t     = np.zeros(n_obs)
    
    for n in range(n_obs):
        v_t[n] = theta[int(context[n])]
        temp_rt = _ddm_trial(v_t[n], theta[2], theta[3], theta[4])
#         while np.abs(temp_rt) < min_rt or np.abs(temp_rt) > max_rt:
#             temp_rt =  _ddm_trial(v_t[n], theta[2], theta[3], theta[4])
        rt[n] = temp_rt
        
    return rt[:, np.newaxis]


def standard_levy_process(theta, context, fix_alpha=False):
    n_obs = context.shape[0]
    rt      = np.zeros(n_obs)
    v_t     = np.zeros(n_obs)
    
    if fix_alpha is False:
        alpha = theta[-1]
    else:
        alpha = fix_alpha
        
    for n in range(n_obs):
        v_t[n] = theta[int(context[n])]
        temp_rt = levy_trial(v_t[n], theta[2], theta[3], theta[4], alpha, 0.001, 100000)
#         while np.abs(temp_rt) < min_rt or np.abs(temp_rt) > max_rt:
#             temp_rt = levy_trial(v_t[n], theta[2], theta[3], theta[4], alpha, 0.001, 100000)
        rt[n] = temp_rt
    
    return rt[:, np.newaxis].astype(np.float32)



def levy_process(theta, context, fix_alpha=False): #  n_obs, add this for variable num of observation 
    """
    v1, v2, v_s, a, bias, bias_s, ndt, ndt_s, alpha = theta
    """

    n_obs = context.shape[0] # n_obs is extracted from the context
    rt    = np.zeros(n_obs)
    v_t   = np.zeros(n_obs)
    
    bias_t = np.random.uniform(theta[4] - theta[5]/2, theta[4] + theta[5]/2, size=int(n_obs))
    ndt_t  = np.random.uniform(theta[6] - theta[7]/2, theta[6] + theta[7]/2, size=int(n_obs))
    
    if fix_alpha is False:
        alpha = theta[8]
    else:
        alpha = fix_alpha
    
    for n in range(n_obs):
        v_t[n] = theta[int(context[n])] + theta[2] * np.random.randn()
        temp_rt  = levy_trial(v_t[n], theta[3], bias_t[n], ndt_t[n], alpha, 0.001, 100000) # the last one is alpha 
#         while np.abs(temp_rt) < min_rt or np.abs(temp_rt) > max_rt:
#             temp_rt = levy_trial(v_t[n], theta[3], bias_t[n], ndt_t[n], alpha, 0.001, 100000)
            
        rt[n] = temp_rt
    
    return rt[:, np.newaxis].astype(np.float32)