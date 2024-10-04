import numpy as np
from scipy import stats
from scipy.stats import truncnorm, levy_stable, halfnorm

RNG = np.random.default_rng(1923)

def ddm_prior():
    """Generates random draws from priors over the full diffusion decision parameters:
    v1, v2, v_s, a, bias, bias_s, ndt, ndt_s.
    
    Parameters:
    ------------
    v1: float,
        The first drift rate parameter.
    v2: float,
        The second drift rate parameter.
    v_s: float,
        The across-trial variability parameter of drift rate.
    a: float,
        The boundary separation parameter.
    bias:
        The starting point parameter.
    bias_s: 
        The across-trial variability parameter.
    ndt:
        The non-decision time parameter.
    ndt_s:
        The across trial variability parameter.
    Returns:
    ---------
    """

    v1   = RNG.normal(2, 2)
    v2   = RNG.normal(-2, 2)
    v_s  = truncnorm.rvs(0,2)
    
    # define boundary separation prior
    low_bound, up_bound, loc, scale = 0, np.inf, 3, 1
    lower = (low_bound - loc) / scale
    upper = (up_bound - loc) / scale
    a = truncnorm.rvs(a=lower, 
                             b=upper, 
                             loc=loc,
                             scale=scale)

    bias   = RNG.beta(5, 5)                                   
    bias_s = RNG.beta(1, 4) 

    ndt    = RNG.gamma(3, 1/12)
    ndt_s  = np.random.beta(1, 2)/2              

    return np.hstack([v1, v2, v_s, a, bias, bias_s, ndt, ndt_s])


def standard_ddm_prior():
    """Generates random draws from priors over the standard diffusion decision parameters:
    v1, v2, a, bias, ndt.
    
    Parameters:
    ------------
    v1: float,
        The first drift rate parameter.
    v2: float,
        The second drift rate parameter.
    a:  float,
        The boundary separation parameter.
    bias:
        The starting point parameter.
    ndt:
        The non-decision time parameter.
    Returns:
    ---------
    """
    v1   = RNG.normal(2, 2)
    v2   = RNG.normal(-2, 2)
    
    # define boundary separation prior
    low_bound, up_bound, loc, scale = 0, np.inf, 3, 1
    lower = (low_bound - loc) / scale
    upper = (up_bound - loc) / scale
    a = truncnorm.rvs(a=lower, 
                             b=upper, 
                             loc=loc,
                             scale=scale)

    bias   = RNG.beta(5, 5) 
    ndt    = RNG.gamma(3, 1/12)    

    return np.hstack([v1, v2, a, bias, ndt])


def standard_levy_prior(fix_alpha=False): 
    """Generates random draws from priors over the standard Levy flight parameters:
    v1, v2, a, bias, ndt.
    
    Parameters:
    ------------
    v1: float,
        The first drift rate parameter.
    v2: float,
        The second drift rate parameter.
    a:  float,
        The boundary separation parameter.
    bias:
        The starting point parameter.
    ndt:
        The non-decision time parameter.
    alpha: float,
        The alpha parameter.
    Returns:
    ---------
    """
        
    v1   = RNG.normal(2, 2)
    v2   = RNG.normal(-2, 2)
    
    # define boundary separation prior
    low_bound, up_bound, loc, scale = 0, np.inf, 3, 1
    lower = (low_bound - loc) / scale
    upper = (up_bound - loc) / scale
    a = truncnorm.rvs(a=lower, 
                             b=upper, 
                             loc=loc,
                             scale=scale)
    bias   = RNG.beta(5, 5) 
    ndt    = RNG.gamma(3, 1/12)   
    
    if fix_alpha is False:
        alpha  = RNG.uniform(1, 2)
    else:
        alpha = fix_alpha

    return np.hstack([v1, v2, a, bias, ndt, alpha]) 

def levy_prior(fix_alpha=False):
    """Generates random draws from priors over the Levy flight parameters:
    v1, v2, v_s, a, bias, bias_s, ndt, ndt_s.
    
    Parameters:
    ------------
    v1: float,
        The first drift rate parameter.
    v2: float,
        The second drift rate parameter.
    v_s: float,
        The across-trial variability parameter of drift rate.
    a: float,
        The boundary separation parameter.
    bias:
        The starting point parameter.
    bias_s: 
        The across-trial variability parameter.
    ndt:
        The non-decision time parameter.
    ndt_s:
        The across trial variability parameter.
    alpha: float,
        The alpha parameter.
    Returns:
    ---------
    """
    v1   = RNG.normal(2, 2)
    v2   = RNG.normal(-2, 2)
    v_s  = truncnorm.rvs(0,2)
    
    # define boundary separation prior
    low_bound, up_bound, loc, scale = 0, np.inf, 3, 1
    lower = (low_bound - loc) / scale
    upper = (up_bound - loc) / scale
    a = truncnorm.rvs(a=lower, 
                             b=upper, 
                             loc=loc,
                             scale=scale)
    
    bias   = RNG.beta(5, 5)   
    bias_s = RNG.beta(1, 4) 
       
    ndt    = RNG.gamma(3, 1/12)  
    ndt_s  = np.random.beta(1, 2)/ 2
    
    if fix_alpha is False:
        alpha  = RNG.uniform(1, 2)
    else:
        alpha = fix_alpha
    

    return np.hstack([v1, v2, v_s, a, bias, bias_s, ndt, ndt_s, alpha]) 