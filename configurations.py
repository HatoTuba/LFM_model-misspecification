model_names = ("standard_ddm", "full_ddm", "standard_levy", "full_levy")

def get_params(model_name):
    if model_name=='standard_ddm':
#         param_names= [r'$v_1$', r'$v_2$', r'$a$', r'$z$', r'$ndt$']
        param_names = [r'Drift rate ($v_1$)', r'Drift rate ($v_2$)', r'Boundary ($a$)', 
               r'Starting point ($z$)', r'Non-decision time ($t_0$)']
    if model_name=='standard_levy':
#         param_names= [r'$v_1$', r'$v_2$', r'$a$', r'$z$', r'$ndt$', r'$\alpha$']
        param_names = [r'Drift rate ($v_1$)', r'Drift rate ($v_2$)', r'Boundary ($a$)', 
               r'Starting point ($z$)', r'Non-decision time ($t_0$)', r'Alpha ($\alpha$)']
    if model_name=='full_ddm':
#         param_names= [r'$v_1$', r'$v_2$', r'$v_s$', r'$a$', r'$z$', r'$z_s$', r'$ndt$', r'$ndt_s$']
        param_names = [r'Drift rate ($v_1$)', r'Drift rate ($v_2$)', r'Drift rate var. ($s_v$)', 
                       r'Boundary ($a$)', r'Starting point ($z$)', r'Starting point var. ($s_z$)', 
                       r'Non-decision time ($t_0$)', r'Non-decision time var. ($s_{t_0}$)']
    if model_name=='full_levy':
#         param_names= [r'$v_1$', r'$v_2$', r'$v_s$', r'$a$', r'$z$', r'$z_s$', r'$ndt$', r'$ndt_s$', r'$\alpha$']
        param_names = [r'Drift rate ($v_1$)', r'Drift rate ($v_2$)', r'Drift rate var. ($s_v$)', 
                       r'Boundary ($a$)', 'Starting point ($z$)', r'Starting point var. ($s_z$)', 
                       r'Non-decision time ($t_0$)', r'Non-decision time var. ($s_{t_0}$)', r'Alpha ($\alpha$)']
    return param_names



def configure_splitting(raw_dict):
    rt = np.array_split(raw_dict['sim_data'][:, :, :1].astype(np.float32), 10)
    context = np.array_split(np.array(raw_dict['sim_batchable_context'])[:, :, np.newaxis], 10)
    summary_conditions = np.c_[rt, context].astype(np.float32)
    parameters = np.array_split(raw_dict['prior_draws'].astype(np.float32), 10)
    return parameters, summary_conditions