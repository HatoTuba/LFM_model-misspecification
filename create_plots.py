import os, sys
sys.path.append("../")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import bayesflow as bf
import pickle  
from get_setup import get_setup
from configurations import get_params
from scipy.stats import median_abs_deviation
from sklearn.metrics import r2_score

from bayesflow.helper_functions import check_posterior_prior_shapes
from bayesflow.computational_utilities import expected_calibration_error, simultaneous_ecdf_bands


def plot_recovery( # copied directly from bayesflow
    post_samples,
    prior_samples,
    point_agg=np.median,
    uncertainty_agg=median_abs_deviation,
    param_names=None,
    fig_size=None,
    label_fontsize=22,
    title_fontsize=22,
    metric_fontsize=18,
    tick_fontsize=16,
    add_corr=True,
    add_r2=True,
    color="#156082",
    n_col=None,
    n_row=None,
#     xlabel=None,
#     ylabel=None,
    xlabel="Ground truth",
    ylabel="Estimated",
    **kwargs,
):
    """Creates and plots publication-ready recovery plot with true vs. point estimate + uncertainty.
    The point estimate can be controlled with the ``point_agg`` argument, and the uncertainty estimate
    can be controlled with the ``uncertainty_agg`` argument.

    This plot yields similar information as the "posterior z-score", but allows for generic
    point and uncertainty estimates:

    https://betanalpha.github.io/assets/case_studies/principled_bayesian_workflow.html

    Important: Posterior aggregates play no special role in Bayesian inference and should only
    be used heuristically. For instance, in the case of multi-modal posteriors, common point
    estimates, such as mean, (geometric) median, or maximum a posteriori (MAP) mean nothing.

    Parameters
    ----------
    post_samples      : np.ndarray of shape (n_data_sets, n_post_draws, n_params)
        The posterior draws obtained from n_data_sets
    prior_samples     : np.ndarray of shape (n_data_sets, n_params)
        The prior draws (true parameters) obtained for generating the n_data_sets
    point_agg         : callable, optional, default: ``np.median``
        The function to apply to the posterior draws to get a point estimate for each marginal.
        The default computes the marginal median for each marginal posterior as a robust
        point estimate.
    uncertainty_agg   : callable or None, optional, default: scipy.stats.median_abs_deviation
        The function to apply to the posterior draws to get an uncertainty estimate.
        If ``None`` provided, a simple scatter using only ``point_agg`` will be plotted.
    param_names       : list or None, optional, default: None
        The parameter names for nice plot titles. Inferred if None
    fig_size          : tuple or None, optional, default : None
        The figure size passed to the matplotlib constructor. Inferred if None.
    label_fontsize    : int, optional, default: 16
        The font size of the y-label text
    title_fontsize    : int, optional, default: 18
        The font size of the title text
    metric_fontsize   : int, optional, default: 16
        The font size of the goodness-of-fit metric (if provided)
    tick_fontsize     : int, optional, default: 12
        The font size of the axis tick labels
    add_corr          : bool, optional, default: True
        A flag for adding correlation between true and estimates to the plot
    add_r2            : bool, optional, default: True
        A flag for adding R^2 between true and estimates to the plot
    color             : str, optional, default: '#8f2727'
        The color for the true vs. estimated scatter points and error bars
    n_row             : int, optional, default: None
        The number of rows for the subplots. Dynamically determined if None.
    n_col             : int, optional, default: None
        The number of columns for the subplots. Dynamically determined if None.
    xlabel            : str, optional, default: 'Ground truth'
        The label on the x-axis of the plot
    ylabel            : str, optional, default: 'Estimated'
        The label on the y-axis of the plot
    **kwargs          : optional
        Additional keyword arguments passed to ax.errorbar or ax.scatter.
        Example: `rasterized=True` to reduce PDF file size with many dots

    Returns
    -------
    f : plt.Figure - the figure instance for optional saving

    Raises
    ------
    ShapeError
        If there is a deviation from the expected shapes of ``post_samples`` and ``prior_samples``.
    """

    # Sanity check
    check_posterior_prior_shapes(post_samples, prior_samples)

    # Compute point estimates and uncertainties
    est = point_agg(post_samples, axis=1)
    if uncertainty_agg is not None:
        u = uncertainty_agg(post_samples, axis=1)

    # Determine n params and param names if None given
    n_params = prior_samples.shape[-1]
    if param_names is None:
        param_names = [f"$\\theta_{{{i}}}$" for i in range(1, n_params + 1)]

    # Determine number of rows and columns for subplots based on inputs
    if n_row is None and n_col is None:
        n_row = int(np.ceil(n_params / 6))
        n_col = int(np.ceil(n_params / n_row))
    elif n_row is None and n_col is not None:
        n_row = int(np.ceil(n_params / n_col))
    elif n_row is not None and n_col is None:
        n_col = int(np.ceil(n_params / n_row))

    # Initialize figure
    if fig_size is None:
        fig_size = (int(4 * n_col), int(4 * n_row))
    f, axarr = plt.subplots(n_row, n_col, figsize=fig_size)
    
    # turn axarr into 1D list
    axarr = np.atleast_1d(axarr)
    if n_col > 1 or n_row > 1:
        axarr_it = axarr.flat
    else:
        axarr_it = axarr

    for i, ax in enumerate(axarr_it):
        if i >= n_params:
            break

        # Add scatter and error bars
        if uncertainty_agg is not None:
            _ = ax.errorbar(prior_samples[:, i], est[:, i], yerr=u[:, i], fmt="o", alpha=0.5, color=color, **kwargs)
        else:
            _ = ax.scatter(prior_samples[:, i], est[:, i], alpha=0.5, color=color, **kwargs)

        # Make plots quadratic to avoid visual illusions
        lower = min(prior_samples[:, i].min(), est[:, i].min())
        upper = max(prior_samples[:, i].max(), est[:, i].max())
        eps = (upper - lower) * 0.1
        ax.set_xlim([lower - eps, upper + eps])
        ax.set_ylim([lower - eps, upper + eps])
        ax.plot(
            [ax.get_xlim()[0], ax.get_xlim()[1]],
            [ax.get_ylim()[0], ax.get_ylim()[1]],
            color="black",
            alpha=0.9,
            linestyle="dashed",
        )

        # Add optional metrics and title
        if add_r2:
            r2 = r2_score(prior_samples[:, i], est[:, i])
            if len(param_names) <= 10: # for diagnostics
                ax.text(
                    0.1,
                    0.9,
                    "$R^2$ = {:.3f}".format(r2),
                    horizontalalignment="left",
                    verticalalignment="center",
                    transform=ax.transAxes,
                    size=metric_fontsize,
                )
            else:
                ax.text(
                    0.95,
                    0.2,
                    "$R^2$ = {:.3f}".format(r2),
                    horizontalalignment="right",
                    verticalalignment="top",
                    transform=ax.transAxes,
                    size=metric_fontsize,
                    )
        if add_corr:
            corr = np.corrcoef(prior_samples[:, i], est[:, i])[0, 1]
            if len(param_names) <= 10: # for diagnostics
                ax.text(
                    0.1,
                    0.8,
                    "$r$ = {:.3f}".format(corr),
                    horizontalalignment="left",
                    verticalalignment="center",
                    transform=ax.transAxes,
                    size=metric_fontsize,
                )
            else:
                ax.text(
                    0.87,
                    0.1,
                    "$r$ = {:.3f}".format(corr),
                    horizontalalignment="right",
                    verticalalignment="top",
                    transform=ax.transAxes,
                    size=metric_fontsize,
                    )
        ax.set_title(param_names[i], fontsize=title_fontsize)

        # Prettify
        sns.despine(ax=ax)
        ax.grid(alpha=0.35)
        ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
        ax.tick_params(axis="both", which="minor", labelsize=tick_fontsize)

    # Only add x-labels to the bottom row
    bottom_row = axarr if n_row == 1 else axarr[0] if n_col == 1 else axarr[n_row - 1, :]
    for _ax in bottom_row:
        _ax.set_xlabel(xlabel, fontsize=label_fontsize)

    # Only add y-labels to right left-most row
    if n_row == 1:  # if there is only one row, the ax array is 1D
        axarr[0].set_ylabel(ylabel, fontsize=label_fontsize)
    # If there is more than one row, the ax array is 2D
    else:
        for _ax in axarr[:, 0]:
            _ax.set_ylabel(ylabel, fontsize=label_fontsize)
        # Remove unused axes entirely
    for _ax in axarr_it[n_params:]:
        _ax.set_visible(False)  # Hide the unused axes
        _ax.set_frame_on(False)  # Remove the frame
        _ax.set_facecolor('white')

#     # Remove unused axes entirely
    for _ax in axarr_it[n_params:]:
        _ax.remove()

    f.tight_layout()
    return f
########################################3
def plot_sbc_ecdf(
    post_samples,
    prior_samples,
    difference=False,
    stacked=False,
    fig_size=None,
    param_names=None,
    label_fontsize=22,
    legend_fontsize=24,
    title_fontsize=24,
    tick_fontsize=20,
    rank_ecdf_color="#a34f4f",
    fill_color="grey",
    n_row=None,
    n_col=None,
    **kwargs,
):
    # Sanity checks
    check_posterior_prior_shapes(post_samples, prior_samples)

    # Store reference to number of parameters
    n_params = post_samples.shape[-1]

    # Compute fractional ranks (using broadcasting)
    ranks = np.sum(post_samples < prior_samples[:, np.newaxis, :], axis=1) / post_samples.shape[1]

    # Prepare figure
    if stacked:
        n_row, n_col = 1, 1
        f, ax = plt.subplots(1, 1, figsize=fig_size)
    else:
        if n_row is None and n_col is None:
            n_row = int(np.ceil(n_params / 6))
            n_col = int(np.ceil(n_params / n_row))
        elif n_row is None:
            n_row = int(np.ceil(n_params / n_col))
        elif n_col is None:
            n_col = int(np.ceil(n_params / n_row))

        if fig_size is None:
            fig_size = (int(5 * n_col), int(5 * n_row))

        f, ax = plt.subplots(n_row, n_col, figsize=fig_size)
        ax = np.atleast_1d(ax)

    # List to store legend handles and labels
    handles, labels = [], []

    # Plot individual ECDF of parameters
    for j in range(ranks.shape[-1]):
        ecdf_single = np.sort(ranks[:, j])
        xx = ecdf_single
        yy = np.arange(1, xx.shape[-1] + 1) / float(xx.shape[-1])

        if difference:
            yy -= xx

        if stacked:
            if j == 0:
                plot, = ax.plot(xx, yy, color=rank_ecdf_color, alpha=0.95, label="Rank ECDFs")
            else:
                ax.plot(xx, yy, color=rank_ecdf_color, alpha=0.95)
        else:
            plot, = ax.flat[j].plot(xx, yy, color=rank_ecdf_color, alpha=0.95)
        
        # Collect handles and labels from the first plot only
        if j == 0:
            handles.append(plot)
            labels.append("Rank ECDF")

    # Compute uniform ECDF and bands
    alpha, z, L, H = simultaneous_ecdf_bands(post_samples.shape[0], **kwargs.pop("ecdf_bands_kwargs", {}))

    if difference:
        L -= z
        H -= z
        ylab = "ECDF difference"
    else:
        ylab = "ECDF"

    if stacked:
        titles = [None]
        axes = [ax]
    else:
        axes = ax.flat
        titles = param_names if param_names else [f"$\\theta_{{{i}}}$" for i in range(1, n_params + 1)]

    # Add simultaneous bounds and prettify plots
    for _ax, title in zip(axes, titles):
        fill = _ax.fill_between(z, L, H, color=fill_color, alpha=0.2)
        
        sns.despine(ax=_ax)
        _ax.grid(alpha=0.35)
        _ax.set_title(title, fontsize=title_fontsize)
        _ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
        _ax.tick_params(axis="both", which="minor", labelsize=tick_fontsize)

        # Collect fill_between handle and label only once
        if stacked and title is None or title == titles[0]:
            handles.append(fill)
            labels.append(rf"{int((1 - alpha) * 100)}$\%$ Confidence Bands")

    # Add x-labels to the bottom row
    bottom_row = [ax] if stacked else (ax if n_row == 1 else ax[-1, :])
    for _ax in bottom_row:
        _ax.set_xlabel("Fractional rank statistic", fontsize=label_fontsize)

    # Add y-labels to left-most row
    if n_row == 1:
        axes[0].set_ylabel(ylab, fontsize=label_fontsize)
    else:
        for _ax in ax[:, 0]:
            _ax.set_ylabel(ylab, fontsize=label_fontsize)

    # Remove unused axes
    for _ax in axes[n_params:]:
        _ax.remove()

    # Add the legend at the bottom of the figure
    f.legend(handles, labels, fontsize=legend_fontsize, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=4) #it was ncol=4 for the standard models

    f.tight_layout(rect=[0, 0.03, 1, 0.99])
    return f

############################        
def diagnostic_plots(models): 
    """This function plots all the diagnostics to check network models."""
    for model_name in models:
        exp = get_setup(model_name=model_name)
        diag_sim = exp.model.generate(2000) #2000
        diag_sim = exp.model.configure(diag_sim)
        diag_priors = diag_sim['parameters']
        
        diag_posteriors = exp.amortizer.sample(diag_sim, n_samples=250) #250
        
        param_names = get_params(model_name)
                
#         f = bf.diagnostics.plot_sbc_histograms(diag_posteriors, diag_priors,
#                                               num_bins=10, param_names=param_names)#, color='#033960c')

        d = plot_sbc_ecdf(diag_posteriors, diag_priors, stacked=False, difference=True,
                                         param_names=param_names, rank_ecdf_color='#156082', n_col=3)
        
        c = plot_recovery(diag_posteriors, diag_priors,
                                         point_agg=np.mean, param_names=param_names, color='#156082', n_col=3)
        
#         save_path1 = f"plots/diag_sbc_hist_{model_name}.pdf"
        save_path2 = f"plots/diag_sbc_ecdf_{model_name}.pdf"  
    
        save_path3 = f"plots/diag_plot_recovery_{model_name}.pdf" 
        
#         f.savefig(save_path1)
        d.savefig(save_path2, dpi=300, bbox_inches='tight')
#     f.savefig("../plots/gpddm_rt_time_series_sub_{}.pdf".format(sub+1), dpi=300, bbox_inches='tight')
        c.savefig(save_path3, dpi=300, bbox_inches='tight')

#         print(f"Plots saved: {save_path1}, {save_path2}, {save_path3}")
        
###############################
def plot_bias(
    post_samples, # (n_datasets, n_post_draws, n_params) # LF data to DDM
    prior_samples, # (n_datasets, n_params)              # LF data
    stan_samples=None,
    point_agg=np.median,
    uncertainty_agg=np.std,
    param_names=None,
    fig_size=None,
    label_fontsize=27,
    title_fontsize=24,
    metric_fontsize=20,
    tick_fontsize=20, # only for standard models, otherwise set to 18 or 16
    add_corr=True,
    color="#156082",
    n_col=None,
    n_row=None,
):
    
    if param_names is None:
        raise ValueError("param_names cannot be None.")
    if len(param_names) == 0:
        raise ValueError("param_names cannot be empty list.")
    
    prior_samples_lf = None 
    # prepare shapes for plotting
    if prior_samples.shape[1] > len(param_names):
        if stan_samples is None:        
            prior_samples_lf = prior_samples[:, :]    # including alpha values
            prior_samples = prior_samples_lf[:, 0:-1] # eliminate the alpha for plotting
        else:
            prior_samples_lf = prior_samples[0:100, :] # including alpha values
            prior_samples = prior_samples[0:100, 0:-1] # eliminate the alpha for plotting
            
    
    alpha_values = prior_samples_lf[:, -1] if prior_samples_lf is not None else None
    
    # Compute point estimates and uncertainties
    bf_est = point_agg(post_samples, axis=1)
    un_est = uncertainty_agg(prior_samples, axis=0) # try uncertainty of whole true param value
    
    if stan_samples is None:
#         bias = (prior_samples - bf_est) / un_est[np.newaxis, 0:len(param_names)]
        bias = (bf_est - prior_samples) / un_est[np.newaxis, 0:len(param_names)] # (estimated param-ground truth)/std(ground truth)
    else:
        stan_est = point_agg(stan_samples, axis=1) # compare stan and bf
#         bias = (stan_est - bf_est) / un_est[np.newaxis, 0:len(param_names)] 
        bias = (bf_est - stan_est) / un_est[np.newaxis, 0:len(param_names)]  # (estimated param-ground truth)/std(ground truth)


    # Determine n params and param names if None given
    n_params = prior_samples.shape[-1]
    if param_names is None:
        param_names = [f"$p_{i}$" for i in range(1, n_params + 1)]

    
    # Determine number of rows and columns for subplots based on inputs
    if n_row is None and n_col is None:
        n_row = int(np.ceil(n_params / 6))
        n_col = int(np.ceil(n_params / n_row))
    elif n_row is None and n_col is not None:
        n_row = int(np.ceil(n_params / n_col))
    elif n_row is not None and n_col is None:
        n_col = int(np.ceil(n_params / n_row))
        
    
    # Initialize figure
    if fig_size is None:
        fig_size = (int(6 * n_col), int(4 * n_row))
    f, axarr = plt.subplots(n_row, n_col, figsize=fig_size, sharey=True)
    
    # turn axarr into 1D list
    if n_col > 1 or n_row > 1:
        axarr = axarr.flat
    else:
        # for 1x1, axarr is not a list
        axarr = [axarr]
        
    for i, ax in enumerate(axarr):
        if i >= n_params:
            break
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize) 
    
        im = ax.scatter(prior_samples_lf[:, -1],  bias[:, i], alpha=0.5, color=color) 
        sns.regplot(x=prior_samples_lf[:, -1], y=bias[:, i], scatter=False, ax=ax, color='#2E3033')
        
        # Make plots quadratic to avoid visual illusions
        ax.set_ylim(min(bias[:, i]-1), max(bias[:, i]+1))

        # only for standard bias, to illustrate the effects better 
        ax.set_ylim(-2, max(bias[:, i]+1))
#         ax.set_ylim(-1, 1)
        ax.axhline(y=0, linestyle='--', color='darkgray')
        ax.grid(alpha=0.35)
         
#         if i >= 4: # for full models comment out for the standard
#             ax.set_xlabel(r'$\alpha$', fontsize=label_fontsize)
        ax.set_xlabel(r'$\alpha$: Misspecification severity', fontsize=label_fontsize) # comment out for full models
        
        if add_corr:
            corr = np.corrcoef(prior_samples_lf[:, -1], bias[:, i])[0, 1]

            if len(param_names) == 5 or stan_samples is not None:
                ax.text(
                    0.7,
                    0.2,
                    "$r$ = {:.3f}".format(corr),
                    horizontalalignment="left",
                    verticalalignment="center",
                    transform=ax.transAxes,
                    size=metric_fontsize,
                    )
            else:
                ax.text(
                    0.95,
                    0.9,
                    "$r$ = {:.3f}".format(corr),
                    horizontalalignment="right",
                    verticalalignment="top",
                    transform=ax.transAxes,
                    size=metric_fontsize,
                    )
        
        if i == 0:# or i==4:
            ax.set_ylabel('Bias', fontsize=label_fontsize)  

            
        if len(param_names) <= 5 and i == 0:
            ax.set_ylabel('Bias', fontsize=label_fontsize)
            ax.set_xlabel(r'$\alpha$: Misspecification severity', fontsize=label_fontsize)

        elif len(param_names) > 6 and i in (4, 5, 6,7):
#             ax.set_ylabel('Bias', fontsize=label_fontsize)
            ax.set_xlabel(r'$\alpha$: Misspecification severity', fontsize=label_fontsize)

        ax.set_title(param_names[i], fontsize=title_fontsize)
        
    # Prettify
    
        sns.despine(ax=ax)
        ax.grid(alpha=0.5)
    f.tight_layout()
    f.subplots_adjust(hspace=0.5)

    
    
    if len(param_names)==5 and stan_samples is None:
        save_path = f"plots/absolute_bias_simple.pdf"
    elif len(param_names)==5:
        save_path = f"plots/absolute_bias_stanvsbf.pdf"
    else:
        save_path = f"plots/absolute_bias_full.pdf"
    f.savefig(save_path)
        
###########################     
def compare_estimates(
    samples_x,
    samples_y,
    point_agg=np.median,
    uncertainty_agg=np.std,
    param_names=None,
    fig_size=None,
    label_x="x",
    label_y="y",
    label_fontsize=22,
    legend_fontsize=22,
    title_fontsize=22,
    tick_fontsize=16,
    metric_fontsize=20,
    add_corr=True,
    add_r2=True,
    color="#156082",
    markersize=6.0,
    n_col=None,
    n_row=None,
):
    """Creates and plots publication-ready plot with point estimates + uncertainty.
    The point estimates can be controlled with the `point_agg` argument, and the uncertainty estimates
    can be controlled with the `uncertainty_agg` argument.
    Important: Posterior aggregates play no special role in Bayesian inference and should only
    be used heuristically. For instance, in the case of multi-modal posteriors, common point
    estimates, such as mean, (geometric) median, or maximum a posteriori (MAP) mean nothing.
    Parameters
    ----------
    samples_x         : np.ndarray of shape (n_data_sets, n_post_draws, n_params)
        The first set of posterior draws obtained from n_data_sets
    samples_y         : np.ndarray of shape (n_data_sets, n_post_draws, n_params)
        The second set of posterior draws obtained from n_data_sets
    point_agg         : callable, optional, default: np.mean
        The function to apply to the posterior draws to get a point estimate for each marginal.
    uncertainty_agg   : callable or None, optional, default: np.std
        The function to apply to the posterior draws to get an uncertainty estimate.
        If `None` provided, a simple scatter will be plotted.
    param_names       : list or None, optional, default: None
        The parameter names for nice plot titles. Inferred if None
    fig_size          : tuple or None, optional, default : None
        The figure size passed to the matplotlib constructor. Inferred if None.
    label_x           : string, optional, default: 'x'
        The x-label text
    label_y           : string, optional, default: 'y'
        The y-label text
    label_fontsize    : int, optional, default: 14
        The font size of the y-label text
    title_fontsize    : int, optional, default: 16
        The font size of the title text
    metric_fontsize   : int, optional, default: 16
        The font size of the goodness-of-fit metric (if provided)
    add_corr          : boolean, optional, default: True
        A flag for adding correlation between true and estimates to the plot.
    add_r2            : boolean, optional, default: True
        A flag for adding R^2 between true and estimates to the plot.
    color             : str, optional, default: '#8f2727'
        The color for the true vs. estimated scatter points and error bars.
    markersize        : float, optional, default: 6.0
        The marker size in points.

    Returns
    -------
    f : plt.Figure - the figure instance for optional saving
    """

    # Compute point estimates and uncertainties
    est_x = point_agg(samples_x, axis=1)
    est_y = point_agg(samples_y, axis=1)
    if uncertainty_agg is not None:
        u_x = uncertainty_agg(samples_x, axis=1)
        u_y = uncertainty_agg(samples_y, axis=1)

    # Determine n params and param names if None given
    n_params = samples_x.shape[-1]
    if param_names is None:
        param_names = [f"p_{i}" for i in range(1, n_params + 1)]

    # Determine number of rows and columns for subplots based on inputs
    if n_row is None and n_col is None:
        n_row = int(np.ceil(n_params / 6))
        n_col = int(np.ceil(n_params / n_row))
    elif n_row is None and n_col is not None:
        n_row = int(np.ceil(n_params / n_col))
    elif n_row is not None and n_col is None:
        n_col = int(np.ceil(n_params / n_row))

    # Initialize figure
    if fig_size is None:
        fig_size = (int(4 * n_col), int(4 * n_row))
    f, axarr = plt.subplots(n_row, n_col, figsize=fig_size)

    # turn axarr into 1D list
    if n_col > 1 or n_row > 1:
        axarr = axarr.flat
    else:
        # for 1x1, axarr is not a list -> turn it into one for use with enumerate
        axarr = [axarr]

    for i, ax in enumerate(axarr):
        if i >= n_params:
            ax.set_visible(False)  # Hide the unused axes
            ax.set_frame_on(False)  # Remove the frame
            ax.set_facecolor('white')  # Set background to white
#             ax.tick_params(axis='both', which='major', labelsize=tick_fontsize) 

            continue

        # Add scatter and errorbars
        if uncertainty_agg is not None:
            if len(u_x.shape) == 3:
                im = ax.errorbar(
                    est_x[:, i],
                    est_y[:, i],
                    xerr=u_x[:, :, i],
                    yerr=u_y[:, :, i],
                    fmt="o",
                    alpha=0.5,
                    color=color,
                    markersize=markersize,
                )
            else:
                im = ax.errorbar(
                    est_x[:, i],
                    est_y[:, i],
                    xerr=u_x[:, i],
                    yerr=u_y[:, i],
                    fmt="o",
                    alpha=0.5,
                    color=color,
                    markersize=markersize,
                )
        else:
            im = ax.scatter(
                est_x[:, i], est_y[:, i], alpha=0.5, color=color, s=markersize**2
            )

        # Make plots quadratic to avoid visual illusions
        lower = min(est_x[:, i].min(), est_y[:, i].min())
        upper = max(est_x[:, i].max(), est_y[:, i].max())
        eps = (upper - lower) * 0.1
        ax.set_xlim([lower - eps, upper + eps])
        ax.set_ylim([lower - eps, upper + eps])
        ax.plot(
            [ax.get_xlim()[0], ax.get_xlim()[1]],
            [ax.get_ylim()[0], ax.get_ylim()[1]],
            color="black",
            alpha=0.9,
            linestyle="dashed",
        )

        # Add labels, optional metrics and title
        ax.set_xlabel(label_x, fontsize=label_fontsize)
        ax.set_ylabel(label_y, fontsize=label_fontsize)
        if add_r2:
            r2 = r2_score(est_x[:, i], est_y[:, i])
            ax.text(
                0.1,
                0.9,
                "$R^2$ = {:.3f}".format(r2),
                horizontalalignment="left",
                verticalalignment="center",
                transform=ax.transAxes,
                size=metric_fontsize,
            )
        if add_corr:
            corr = np.corrcoef(est_x[:, i], est_y[:, i])[0, 1]
            ax.text(
                0.1,
                0.8,
                "$r$ = {:.3f}".format(corr),
                horizontalalignment="left",
                verticalalignment="center",
                transform=ax.transAxes,
                size=metric_fontsize,
            )
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

        ax.set_title(param_names[i], fontsize=title_fontsize)
        # Prettify
        sns.despine(ax=ax)
        ax.grid(alpha=0.35)

    f.tight_layout()
    return f