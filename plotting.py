import argparse
import itertools
import os
import string

import matplotlib.pyplot as plt
import tqdm
from matplotlib.gridspec import GridSpec
from matplotlib.colors import BoundaryNorm
import matplotlib.colors as mcolors
from matplotlib.transforms import blended_transform_factory
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle, Polygon

import statsmodels.api as sm
from scipy.optimize import root_scalar

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.metrics import r2_score
from numpy.polynomial.polynomial import polyfit, polyval
import cartopy.crs as ccrs

from lib import get_country_name_dicts
from lib_compute_resilience_and_risk import agg_to_event_level
from lib_prepare_scenario import average_over_rp
from prepare_scenario import load_income_groups
from recovery_optimizer import baseline_consumption_c_h, delta_c_h_of_t, delta_k_h_eff_of_t
import seaborn as sns

INCOME_GROUP_COLORS = {
    'LICs': plt.get_cmap('tab10')(0),
    'LMICs': plt.get_cmap('tab10')(1),
    'UMICs': plt.get_cmap('tab10')(2),
    'HICs': plt.get_cmap('tab10')(3),
}

HAZARD_COLORS = {
    'Earthquake': plt.get_cmap('tab10')(0),
    'Flood': plt.get_cmap('tab10')(1),
    'Tsunami': plt.get_cmap('tab10')(2),
    'Storm surge': plt.get_cmap('tab10')(3),
    'Wind': plt.get_cmap('tab10')(4),
}

INCOME_GROUP_MARKERS = ['o', 's', 'D', '^']

NAME_DICT = {
    'resilience': 'socio-economic\nresilience [%]',
    'risk': 'risk to wellbeing\n[% of GDP]',
    'risk_to_assets': 'risk to assets\n[% of GDP]',
    'gdp_pc_pp': 'GDPpc [$PPP]',
    'log_gdp_pc_pp': 'ln(GDPpc [$PPP])',
    'dk_tot': 'risk to assets\n[$PPP]',
    'dWtot_currency': 'risk to wellbeing\n[$PPP]',
    'gini_index': 'Gini index [%]',
}

# Set the default font size for labels and ticks
plt.rcParams['axes.labelsize'] = 7  # Font size for x and y labels
plt.rcParams['xtick.labelsize'] = 6  # Font size for x tick labels
plt.rcParams['ytick.labelsize'] = 6  # Font size for y tick labels
plt.rcParams['legend.fontsize'] = 6  # Font size for legend
plt.rcParams['axes.titlesize'] = 7  # Font size for axis title
plt.rcParams['font.size'] = 7  # Font size for text

# figure widths
# "Column-and-a-Half: 120–136 mm wide"
single_col_width = 8.9  # cm
double_col_width = 18.3  # cm
max_fig_height = 24.7  # cm

inch = 2.54
centimeter = 1 / inch


def format_axis(ax, x_name=None, y_name=None, name_mapping=None, title='infer', ylim=None, xlim=None):
    if name_mapping is not None:
        x_name = name_mapping.get(x_name, x_name)
        y_name = name_mapping.get(y_name, y_name)
    if title == 'infer':
        title = f'{y_name} vs {x_name}'
    else:
        if name_mapping is not None:
            title = name_mapping.get(title, None)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(ylim)
    if xlim:
        ax.set_xlim(xlim)


def add_regression(ax_, data_, x_var_, y_var_, p_val_pos='above left'):
    # Perform OLS regression
    reg_data = data_.sort_values(by=x_var_).copy()
    X = sm.add_constant(reg_data[x_var_])
    y = reg_data[y_var_]
    model = sm.OLS(y, X).fit()
    print(model.summary())

    # Get the regression line
    regline = model.predict(X)

    # Get the confidence intervals
    predictions = model.get_prediction(X)
    pred_int = predictions.conf_int(alpha=0.05)

    # Plot the regression line
    ax_.plot(reg_data[x_var_], regline, color='k', linestyle='--', lw=.75)

    # Plot the confidence intervals
    ax_.fill_between(reg_data[x_var_], pred_int[:, 0], pred_int[:, 1], color='k', alpha=0.15, lw=0)

    # Determine significance level
    p_value = model.pvalues[1]  # p-value for the slope coefficient
    if p_value < 0.001:
        significance = '***'
    elif p_value < 0.01:
        significance = '**'
    elif p_value < 0.05:
        significance = '*'
    else:
        significance = ' (n.s.)'  # not significant

    r2 = model.rsquared

    # Add significance stars to the plot
    if p_val_pos == 'above left':
        xy = (0, 1.01)
        ha = 'left'
        va = 'bottom'
    elif p_val_pos == 'above right':
        xy = (1, 1.01)
        ha = 'left'
        va = 'bottom'
    elif p_val_pos == 'lower left':
        xy = (.01, .01)
        ha = 'left'
        va = 'bottom'
    elif p_val_pos == 'upper left':
        xy = (.01, .99)
        ha = 'left'
        va = 'top'
    else:
        raise ValueError('p_val_pos should be "above left", "above right", "lower left", or "upper left".')
    ax_.annotate(f'R2={r2:.2f}{significance}', xy=xy, xycoords='axes fraction',
                 ha=ha, va=va, fontsize=6)


def plot_supfig_2(cat_info_data_, macro_data_, iso3='HTI', hazard='Earthquake', plot_rp=100, show=False, outfile=None):
    if (cat_info_data_.help_received.unique() != 0).all():
        raise ValueError("Mus pass a data set without PDS.")
    data = pd.merge(
        cat_info_data_,
        macro_data_,
        left_index=True,
        right_index=True,
    )

    plot_data = pd.DataFrame(
        index=['q1', 'q2', 'q3', 'q4', 'q5', 'q_avg'],
    )

    for q in ['q1', 'q2', 'q3', 'q4', 'q5']:
        t_max = data.loc[pd.IndexSlice[iso3, hazard, plot_rp, q, 'a', 'not_helped'], 't_reco_95'].max()
        num_timesteps = int(10000 * t_max)
        dt = 1 / num_timesteps
        t_ = np.linspace(0, dt * num_timesteps, num_timesteps + 1)
        q_data = data.loc[(iso3, hazard, plot_rp, q, 'a', 'not_helped')]
        di_h_lab, di_h_sp, dc_reco, dc_savings_pds = delta_c_h_of_t(
            t_=t_,
            productivity_pi_=q_data.avg_prod_k,
            delta_tax_sp_=q_data.tau_tax,
            delta_k_h_eff_=q_data.dk,
            lambda_h_=q_data.lambda_h,
            sigma_h_=q_data.k_household_share,
            savings_s_h_=q_data.liquidity,
            delta_i_h_pds_=0,
            delta_c_h_max_=np.nan,
            recovery_params_=q_data.recovery_params,
            social_protection_share_gamma_h_=q_data.gamma_SP,
            consumption_floor_xi_=None,
            t_hat=None,
            t_tilde=None,
            delta_tilde_k_h_eff=None,
            consumption_offset=None,
            return_elements=True
        )
        plot_data.loc[q, 'y_bl'] = q_data.k * q_data.avg_prod_k * dt * num_timesteps
        plot_data.loc[q, 'dy'] = di_h_lab.sum() * dt / (1 - q_data.tau_tax)
        plot_data.loc[q, 'y'] = plot_data.loc[q, 'y_bl'] - plot_data.loc[q, 'dy']
        plot_data.loc[q, 'tr_bl'] = plot_data.loc[q, 'y_bl'] * q_data.tau_tax
        plot_data.loc[q, 'dtr'] = plot_data.loc[q, 'dy'] * q_data.tau_tax
        plot_data.loc[q, 'tr'] = plot_data.loc[q, 'y'] * q_data.tau_tax
        plot_data.loc[q, 'c_bl'] = q_data.c * dt * num_timesteps
        plot_data.loc[q, 'dc_savings'] = -dc_savings_pds.sum() * dt
        plot_data.loc[q, 'dc_reco'] = dc_reco.sum() * dt
        plot_data.loc[q, 'dc_income_sp'] = di_h_sp.sum() * dt
        plot_data.loc[q, 'dc_income_lab'] = di_h_lab.sum() * dt
        plot_data.loc[q, 'dc_income'] = (di_h_lab + di_h_sp).sum() * dt
        plot_data.loc[q, 'dc_short_term'] = plot_data.loc[q, ['dc_savings', 'dc_reco', 'dc_income']].sum()
        plot_data.loc[q, 'c'] = plot_data.loc[q, 'c_bl'] - plot_data.loc[q, 'dc_short_term']
        plot_data.loc[q, 'c_income_sp'] = plot_data.loc[q, 'c_bl'] * q_data.diversified_share - plot_data.loc[q, 'dc_income_sp']
        plot_data.loc[q, 'c_income_lab'] = plot_data.loc[q, 'y'] * (1 - q_data.tau_tax)
        plot_data.loc[q, 'c_income'] = plot_data.loc[q, 'c_income_lab'] + plot_data.loc[q, 'c_income_sp']
        plot_data.loc[q, 'dc_long_term'] = q_data.dc_long_term
        plot_data.loc[q, 'dw_long_term'] = q_data.dW_long_term
        plot_data.loc[q, 'dw_short_term'] = q_data.dw - q_data.dW_long_term
        plot_data.loc[q, 'dw'] = q_data.dw
        plot_data.loc[q, 'k'] = q_data.k
        plot_data.loc[q, 'dk'] = plot_data.loc[q, 'k'] * q_data.v_ew
        plot_data.loc[q, 'dk_reco'] = plot_data.loc[q, 'dk'] * q_data.k_household_share
        plot_data.loc[q, 'dk_reco-savings'] = plot_data.loc[q, 'dk_reco'] - plot_data.loc[q, 'dc_savings']

    plot_data.loc['q_avg', ['k', 'y_bl', 'tr_bl', 'c_bl', 'dc_income_sp', 'c_income_sp']] = plot_data[['k', 'y_bl', 'tr_bl', 'c_bl', 'dc_income_sp', 'c_income_sp']].mean()
    plot_data.loc['q_avg', ['dk', 'dk_reco', 'dk_reco-savings', 'dy', 'dtr', 'dc_savings', 'dc_reco', 'dc_income_lab', 'dw_short_term']] = 0
    plot_data.loc['q_avg', 'dc_income'] = plot_data.loc['q_avg', 'dc_income_sp']
    plot_data.loc['q_avg', 'c_income_lab'] = plot_data.loc['q_avg', 'y_bl'] * (1 - macro_data_.tau_tax.loc[iso3, hazard, plot_rp])
    plot_data.loc['q_avg', 'c_income'] = plot_data.loc['q_avg', 'c_income_lab'] + plot_data.loc['q_avg', 'c_income_sp']
    plot_data.loc['q_avg', 'dc_short_term'] = plot_data.loc['q_avg', 'dc_income_sp']
    plot_data.loc['q_avg', 'dc_long_term'] = (plot_data.dc_long_term - plot_data.dc_savings).mean()
    plot_data.loc['q_avg', ['y', 'tr', 'c']] = plot_data.loc['q_avg', ['y_bl', 'tr_bl', 'c_bl']].values - plot_data.loc['q_avg', ['dy', 'dtr', 'dc_short_term']].values
    plot_data.loc['q_avg', ['dw', 'dw_long_term']] = data.loc[pd.IndexSlice[iso3, hazard, plot_rp, :, 'na', 'not_helped'], ['dw', 'dW_long_term']].mean().values

    plot_data[['dk', 'k', 'dk_reco', 'dk_reco-savings']] /= plot_data.iloc[:5].k.sum()
    plot_data[['y_bl', 'y', 'dy', 'tr_bl', 'tr', 'dtr', 'c_bl', 'c', 'dc_short_term', 'dc_savings', 'dc_reco', 'dc_income', 'dc_long_term', 'dc_income_sp', 'dc_income_lab', 'c_income_sp', 'c_income_lab', 'c_income']] /= plot_data.iloc[:5].y_bl.sum()
    # plot_data[['c_bl', 'c', 'dc_short_term', 'dc_savings', 'dc_reco', 'dc_income', 'dc_long_term', 'dc_income_sp', 'dc_income_lab', 'c_income_sp', 'c_income_lab', 'c_income']] /= plot_data.c_bl.sum()
    plot_data[['dw_short_term', 'dw_long_term', 'dw']] /= plot_data.iloc[:5].dw.sum()

    fig_width = double_col_width * centimeter
    fig_height = 0.6 * fig_width
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    x_assets = 0
    x_output = 1 / 6
    x_transfers = 1 / 3
    x_consumption = 1 / 2
    x_wellbeing_loss = 1

    node_width = .025
    node_pad = .02

    nodes = pd.DataFrame(
        columns=['x0', 'y0', 'x1', 'y1', 'label', 'color', 'cum_outflows', 'cum_inflows'],
        index=[
            "a1", "a2", "a3", "a4", "a5", "a_avg", "dk1", "dk2", "dk3", "dk4", "dk5", "dk_avg",
            "dkreco1", "dkreco2", "dkreco3", "dkreco4", "dkreco5", "dkreco_avg",
            "y1", "y2", "y3", "y4", "y5", "y_avg", "dy1", "dy2", "dy3", "dy4", "dy5", "dy_avg",
            "c1", "c2", "c3", "c4", "c5", "c_avg", "dc1", "dc2", "dc3", "dc4", "dc5", "dc_avg",
            "dcreco1", "dcreco2", "dcreco3", "dcreco4", "dcreco5", "dcreco_avg",
            "dcsav1", "dcsav2", "dcsav3", "dcsav4", "dcsav5", "dcsav_avg",
            "w1_st", "w2_st", "w3_st", "w4_st", "w5_st", "w_avg_st",
            "w1_lt", "w2_lt", "w3_lt", "w4_lt", "w5_lt", "w_avg_lt", "t"
        ]
    )
    nodes.loc[["a1", "a2", "a3", "a4", "a5", "a_avg", "dk1", "dk2", "dk3", "dk4", "dk5", "dk_avg", "dkreco1", "dkreco2", "dkreco3", "dkreco4", "dkreco5", "dkreco_avg"], 'x0'] = x_assets - node_width / 2
    nodes.loc[["y1", "y2", "y3", "y4", "y5", "y_avg", "dy1", "dy2", "dy3", "dy4", "dy5", "dy_avg"], 'x0'] = x_output - node_width / 2
    nodes.loc[["c1", "c2", "c3", "c4", "c5", "c_avg", "dc1", "dc2", "dc3", "dc4", "dc5", "dc_avg", "dcreco1", "dcreco2", "dcreco3", "dcreco4", "dcreco5", "dcreco_avg", "dcsav1", "dcsav2", "dcsav3", "dcsav4", "dcsav5", "dcsav_avg"], 'x0'] = x_consumption - node_width / 2
    nodes.loc[["w1_st", "w2_st", "w3_st", "w4_st", "w5_st", "w_avg_st"], 'x0'] = x_wellbeing_loss - node_width
    nodes.loc[["w1_lt", "w2_lt", "w3_lt", "w4_lt", "w5_lt", "w_avg_lt"], 'x0'] = x_wellbeing_loss
    nodes.loc["t", 'x0'] = x_transfers - node_width / 2
    nodes.loc[:, 'x1'] = nodes.loc[:, 'x0'] + node_width
    nodes.loc[:, 'label'] = nodes.index
    nodes.loc[:, 'cum_outflows'] = 0
    nodes.loc[:, 'cum_inflows'] = 0

        # set the node extents
    y1 = np.array([1, 1, 1, 1]).reshape(4)
    for q in [1, 2, 3, 4, 5, '_avg']:
        nodes.loc[[f"a{q}", f"y{q}", f'c{q}', f'w{q}_st'], 'y1'] = y1
        nodes.loc[[f"dk{q}", f"dy{q}", f'dc{q}'], 'y1'] = nodes.loc[[f"a{q}", f"y{q}", f'c{q}'], 'y1'].values
        node_heights = plot_data.loc[f'q{q}', ['k', 'dk', 'y_bl', 'dy', 'c_bl', 'dc_short_term', 'dw_short_term']].values
        nodes.loc[[f"a{q}", f"dk{q}", f"y{q}", f"dy{q}", f'c{q}', f"dc{q}", f'w{q}_st'], 'y0'] = (nodes.loc[[f"a{q}", f"dk{q}", f"y{q}", f"dy{q}", f'c{q}', f"dc{q}", f'w{q}_st'], 'y1'] - node_heights).values
        # y1 = y1 - (node_heights + node_pad)
        nodes.loc[f"w{q}_lt", 'y1'] = nodes.loc[f"w{q}_st", 'y0']
        nodes.loc[f"w{q}_lt", 'y0'] = nodes.loc[f"w{q}_lt", 'y1'] - plot_data.loc[f"q{q}", "dw_long_term"]
        nodes.loc[[f"dkreco{q}", f"dcreco{q}"], 'y0'] = nodes.loc[[f"dk{q}", f"dc{q}"], 'y0'].values
        nodes.loc[f"dkreco{q}", 'y1'] = nodes.loc[f"dkreco{q}", 'y0'] + plot_data.loc[f"q{q}", 'dk_reco']
        nodes.loc[f"dcreco{q}", 'y1'] = nodes.loc[f"dcreco{q}", 'y0'] + plot_data.loc[f"q{q}", 'dc_reco'] + plot_data.loc[f"q{q}", 'dc_savings']
        nodes.loc[f"dcsav{q}", 'y1'] = nodes.loc[f"dcreco{q}", 'y0']
        nodes.loc[f"dcsav{q}", 'y0'] = nodes.loc[f"dcsav{q}", 'y1'] + plot_data.loc[f"q{q}", 'dc_savings']
        # nodes.loc[[f"dkreco{q}", f"dcreco{q}"], 'y1'] = nodes.loc[[f"dkreco{q}", f"dcreco{q}"], 'y0'] + plot_data.loc[f"q{q}", ['dk_reco', 'dc_reco']].values
        y1 = (nodes.loc[[f"a{q}", f"y{q}", f'c{q}', f'w{q}_st'], 'y0'] - node_pad).values

    q_avg_transfers = plot_data.loc['q_avg', 'tr']

    transfer_break_diameter = 0.05
    nodes.loc["t", "y1"] = nodes.loc[['y5', 'c5', 'a5'], 'y0'].min() - 2 * node_pad
    nodes.loc["t", "y0"] = nodes.loc["t", "y1"] - plot_data.drop('q_avg')[['tr', 'dtr']].sum().sum() - q_avg_transfers - transfer_break_diameter

    nodes.loc[['a_avg', 'dk_avg', 'y_avg', 'dy_avg', 'c_avg', 'dc_avg', 'w_avg_st'], 'y1'] = nodes.loc["t", "y0"] - 2 * node_pad
    nodes.loc[['a_avg', 'dk_avg', 'y_avg', 'dy_avg', 'c_avg', 'dc_avg', 'w_avg_st'], 'y0'] = nodes.loc[['a_avg', 'dk_avg', 'y_avg', 'dy_avg', 'c_avg', 'dc_avg', 'w_avg_st'], 'y1'] - plot_data.loc['q_avg', ['k', 'dk', 'y', 'dy', 'c', 'dc_short_term', 'dw_short_term']].values
    nodes.loc[f"w_avg_lt", 'y1'] = nodes.loc[f"w_avg_st", 'y0']
    nodes.loc[f"w_avg_lt", 'y0'] = nodes.loc[f"w_avg_lt", 'y1'] - plot_data.loc[f"q_avg", "dw_long_term"]
    # nodes.loc[['y_avg', 'c_avg'], 'color'] = 'w'

    nodes.loc[:, 'color'] = 'k'
    nodes.loc[:, 'hatch'] = None
    nodes.loc[["dk1", "dk2", "dk3", "dk4", "dk5", "dk_avg", "dy1", "dy2", "dy3", "dy4", "dy5", "dy_avg", "dc1", "dc2", "dc3", "dc4", "dc5", "dc_avg", "w1_st", "w2_st", "w3_st", "w4_st", "w5_st", "w_avg_st", "w1_lt", "w2_lt", "w3_lt", "w4_lt", "w5_lt", "w_avg_lt"], 'color'] = 'grey'
    nodes.loc[["dkreco1", "dkreco2", "dkreco3", "dkreco4", "dkreco5", "dkreco_avg", "dcreco1", "dcreco2", "dcreco3", "dcreco4", "dcreco5", "dcreco_avg", "dcsav1", "dcsav2", "dcsav3", "dcsav4", "dcsav5", "dcsav_avg"], 'color'] = 'grey'
    nodes.loc[["dkreco1", "dkreco2", "dkreco3", "dkreco4", "dkreco5", "dkreco_avg", "dcreco1", "dcreco2", "dcreco3", "dcreco4", "dcreco5", "dcreco_avg"], 'hatch'] = '//////'
    nodes.loc[["dcsav1", "dcsav2", "dcsav3", "dcsav4", "dcsav5", "dcsav_avg"], 'hatch'] = 'XXXXXX'

    transfer_break_height = 0.02
    transfer_break_coords = np.array([
        [nodes.loc['t', 'x0'] - (nodes.loc['t', 'x1'] - nodes.loc['t', 'x0']) * .05, nodes.loc['t', 'y0'] + q_avg_transfers],
        [nodes.loc['t', 'x0'] - (nodes.loc['t', 'x1'] - nodes.loc['t', 'x0']) * .05, nodes.loc['t', 'y0'] + q_avg_transfers + transfer_break_height],
        [nodes.loc['t', 'x1'] + (nodes.loc['t', 'x1'] - nodes.loc['t', 'x0']) * .05, nodes.loc['t', 'y0'] + q_avg_transfers + transfer_break_diameter],
        [nodes.loc['t', 'x1'] + (nodes.loc['t', 'x1'] - nodes.loc['t', 'x0']) * .05, nodes.loc['t', 'y0'] + q_avg_transfers - transfer_break_height + transfer_break_diameter],
    ])

    links = pd.DataFrame(
        columns=['source', 'target', 'left_y0', 'left_y1', 'right_y0', 'right_y1', 'color', 'alpha', 'kernel_size'],
    )

    def add_link(source, target, left, right, color, alpha, kernel_size, left_y1=None, right_y1=None, append_flows=True):
        if left_y1 is None:
            left_y1 = nodes.loc[source, 'y1'] - nodes.loc[source, 'cum_outflows']
        left_y0 = left_y1 - left
        if right_y1 is None:
            right_y1 = nodes.loc[target, 'y1'] - nodes.loc[target, 'cum_inflows']
        right_y0 = right_y1 - right
        if append_flows:
            nodes.loc[source, 'cum_outflows'] += left
            nodes.loc[target, 'cum_inflows'] += right
        links.loc[len(links)] = [source, target, left_y0, left_y1, right_y0, right_y1, color, alpha, kernel_size]

    # set the links
    for q in [1, 2, 3, 4, 5, '_avg']:
        q_data = plot_data.loc[f'q{q}']
        add_link(f"a{q}", f'y{q}', q_data.dk, q_data.dy, 'red', .5, 50)
        add_link(f"a{q}", f'y{q}', (q_data.k - q_data.dk), q_data.y, 'green', .5, 50)
        add_link(f"y{q}", f'c{q}', q_data.dc_income_lab, q_data.dc_income_lab, 'red', .5, 50)
        add_link(f"y{q}", 't', q_data.dtr, q_data.dtr, 'red', .5, 50)
        add_link('t', f'c{q}', q_data.dc_income_sp, q_data.dc_income_sp, 'red', .5, 50)
        if q != '_avg':
            add_link(f"y{q}", f'c{q}', q_data.c_income_lab, q_data.c_income_lab, 'green', .5, 50)
            add_link(f"y{q}", 't', q_data.tr, q_data.tr, 'green', .5, 50)
            add_link('t', f'c{q}', q_data.c_income_sp, q_data.c_income_sp, 'green', .5, 50)
        else:
            add_link(f"y{q}", 't', q_data.tr, q_data.tr, 'green', .5, 50)
            add_link('t', f'c{q}', q_data.c_income_sp, q_data.c_income_sp, 'green', .5, 50)
            add_link(f"y{q}", f'c{q}', q_data.c_income_lab, q_data.c_income_lab, 'green', .5, 50)
        add_link(f"c{q}", f'w{q}_st', q_data.dc_income, q_data.dc_income / q_data.dc_short_term * q_data.dw_short_term, 'red', .5, 50)
        add_link(f"c{q}", f'w{q}_st', q_data.dc_reco + q_data.dc_savings, (q_data.dc_reco + q_data.dc_savings) / q_data.dc_short_term * q_data.dw_short_term, 'red', .25, 50)
        # add_link(f"a{q}", f"c{q}", q_data['dk_reco-savings'], q_data.dc_reco + q_data.dc_savings, 'red', .25, 50, nodes.loc[f"a{q}", "y1"] - (q_data.dk - q_data['dk_reco-savings']), links[(links.source == f'y{q}') & (links.target == f'c{q}') & (links.color == 'green')]['right_y1'].item(), False)
        if q == 5:
            nodes.loc["t", ["cum_outflows", "cum_inflows"]] += transfer_break_diameter

    node_boxes = [Rectangle((row.x0, row.y0), row.x1 - row.x0, row.y1 - row.y0, hatch=row.hatch, facecolor=row.color) for idx, row in nodes.iterrows()]
    node_boxes = node_boxes + [Polygon(transfer_break_coords, closed=True, facecolor='w')]
    # nodes_pc = PatchCollection(node_boxes, facecolors=nodes.color.to_list() + ['w'], edgecolors='none')
    # ax.add_collection(nodes_pc)
    for node_box in node_boxes:
        ax.add_patch(node_box)

    def calc_strip(left_y0, right_y0, left_y1, right_y1, k_size):
        ys_list = []
        for left_y, right_y in [(left_y0, right_y0), (left_y1, right_y1)]:
            ys = np.array(100 * [left_y] + 100 * [right_y])
            ys = np.convolve(ys, (1 / k_size) * np.ones(k_size), mode='valid')
            ys = np.convolve(ys, (1 / k_size) * np.ones(k_size), mode='valid')
            ys_list.append(ys)

        return ys_list[0], ys_list[1]

    for link_idx, link in list(links.iterrows()):
        ys0, ys1 = calc_strip(link.left_y0, link.right_y0, link.left_y1, link.right_y1, link.kernel_size)
        x_start = nodes.loc[link.source, 'x1']
        x_end = nodes.loc[link.target, 'x0']
        ax.fill_between(np.linspace(x_start, x_end, len(ys0)), ys0, ys1, color=link.color, alpha=link.alpha, lw=0)

    ax.set_ylim(nodes.y0.min(), nodes.y1.max())
    ax.set_xlim(nodes.x0.min(), nodes.x1.max())

    for label, x in zip(['Assets', 'Output', 'Diversification', 'Consumption', 'Wellbeing losses\n(short-term / long-term)'], [x_assets, x_output, x_transfers, x_consumption, x_wellbeing_loss]):
        line1 = label.split('\n')[0]
        line2 = label.split('\n')[1] if len(label.split('\n')) > 1 else ''
        y1 = 1.085
        if line2 != '':
            y1 = 1.111
            ax.text(x, 1.05, line2, ha='center', va='top', fontsize=7)
        ax.text(x, y1, line1, ha='center', va='top', fontsize=8, fontweight='bold')


    for node_idx, node_row in nodes.loc[["y1", "y2", "y3", "y4", "y5", "y_avg"]].iterrows():
        q = node_idx[1:].replace('_', '')
        affected_cat = 'na' if node_idx == 'y_avg' else 'a'
        text = "q" + r"$_{" + q + "}^{" + affected_cat + "}$"
        ax.text(- 2 * node_pad, (node_row.y0 + node_row.y1) / 2, text, ha='center', va='center', fontsize=8)

    ax.axis('off')

    inset_rect_width = (x_wellbeing_loss - x_consumption) * .75
    inset_rect_height = .85
    inset_rect_x0 = x_consumption + (x_wellbeing_loss - x_consumption - inset_rect_width) / 2
    # inset_rect_y0 = (1 - inset_rect_height) / 2
    inset_rect_y0 = 1 - inset_rect_height

    # inset_ax.axis('off')
    inset_welfare_box_coords = np.array(
            [[nodes.loc['c1', 'x1'], nodes.loc['c1', 'y0']],
             [nodes.loc['c1', 'x1'], nodes.loc['c1', 'y1']],
             [inset_rect_x0, inset_rect_y0 + inset_rect_height],
             [inset_rect_x0 + inset_rect_width, inset_rect_y0 + inset_rect_height],
             [nodes.loc['w1_st', 'x0'], nodes.loc['w1_st', 'y1']],
             [nodes.loc['w1_st', 'x0'], nodes.loc['w1_st', 'y0']],
             [inset_rect_x0 + inset_rect_width, inset_rect_y0],
             [inset_rect_x0, inset_rect_y0]]
        )
    inset_welfare_box = Polygon(inset_welfare_box_coords, closed=True)
    ax.add_collection(PatchCollection([inset_welfare_box], facecolors='grey', edgecolors='none', alpha=.4))

    inset_rect_box = Rectangle((inset_rect_x0, inset_rect_y0), inset_rect_width, inset_rect_height, edgecolor='k', facecolor='white')
    ax.add_patch(inset_rect_box)

    inset_ax = ax.inset_axes([inset_rect_x0 + .1 * inset_rect_width, inset_rect_y0 + .1 * inset_rect_height, inset_rect_width * .85, inset_rect_height * .85], transform=ax.transData)

    # hide spines
    inset_ax.spines['top'].set_visible(False)
    inset_ax.spines['right'].set_visible(False)
    # inset_ax.spines['bottom'].set_visible(False)
    # inset_ax.spines['left'].set_visible(False)

    inset_plot_data = pd.merge(
        cat_info_data_,
        macro_data_,
        left_index=True,
        right_index=True,
    ).loc[pd.IndexSlice[iso3, hazard, plot_rp, 'q1', 'a', 'not_helped']]

    plot_recovery(20, inset_plot_data.avg_prod_k, inset_plot_data.tau_tax, inset_plot_data.k, inset_plot_data.dk,
                  inset_plot_data.lambda_h, inset_plot_data.k_household_share, inset_plot_data.liquidity,
                  0, np.nan, inset_plot_data.recovery_params, inset_plot_data.gamma_SP * inset_plot_data.n,
                  inset_plot_data.diversified_share, axs=[inset_ax], show_ylabel=True, plot_capital=False,
                  ylims=[(0, inset_plot_data.c * 1.05), None], plot_legend=False)

    for text, xy, xytext in zip(['Income loss', 'Reconstruction loss', 'Liquidity'], [(0.2, 0.6), (0.145, 0.4), (0.09, 0.21)], [(0.55, 0.4), (0.5, 0.25), (0.45, 0.1)]):
        inset_ax.annotate(
            text=text,
            xy=xy,
            xycoords='axes fraction',
            xytext=xytext,
            textcoords='axes fraction',
            arrowprops=dict(facecolor='black', linewidth=.5, arrowstyle='->'),
            ha='left',
            va='bottom',
            fontsize=7,
        )

    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    inset_ax.set_xticklabels([])
    inset_ax.set_yticklabels([])
    inset_ax.set_ylabel('Consumption')
    inset_ax.set_xlabel('Time')

    plt.tight_layout()

    if outfile:
        plt.savefig(outfile, dpi=300)

    if show:
        plt.show(block=False)


def plot_fig_2(data_, world_, exclude_countries=None, bins_list=None, cmap='viridis', outfile=None,
               show=False, numbering=True, annotate=None, run_ols=False, log_xaxis=False):
    """
    Plots a map with the given data and variables.

    Parameters:
    data_ (str or DataFrame or Series or GeoDataFrame): The data to plot. Can be a path to a CSV file, a pandas DataFrame,
                                                        a pandas Series, or a GeoDataFrame.
    exclude_countries (list or sftr, optional): Countries to exclude from the plot. Defaults to None.
    bins_list (dict, optional): Bins for the variables. Defaults to None.
    cmap (str or dict, optional): Colormap for the variables. Defaults to 'viridis'.
    outfile (str, optional): Path to save the plot. Defaults to None.
    show (bool, optional): Whether to show the plot. Defaults to False.
    show_legend (bool, optional): Whether to show the legend. Defaults to True.
    numbering (bool, optional): Whether to add numbering to the subplots. Defaults to True.

    Returns:
    list: List of axes objects.
    """

    variables = ['risk_to_assets', 'risk', 'resilience']

    # Merge data with world map
    data_ = gpd.GeoDataFrame(pd.merge(data_, world_, on='iso3', how='inner'))
    if len(data_) != len(data_):
        print(f'{len(data_) - len(data_)} countries were not found in the world map. They will be excluded from the plot.')
    data_.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Prepare variables and colormap
    if isinstance(variables, str):
        variables = [variables]
    elif variables is None:
        if isinstance(data_, pd.Series):
            data_ = data_.to_frame()
        variables = list(set(data_.columns) - set(world_.columns) - {'iso3', 'country'})
    if bins_list is None:
        bins_list = {v: None for v in variables}
    if isinstance(cmap, str):
        cmap = {v: cmap for v in variables}
    elif not isinstance(cmap, dict):
        raise ValueError('cmap should be a string or a dictionary.')

    # Exclude specified countries
    if exclude_countries:
        if isinstance(exclude_countries, str):
            exclude_countries = [exclude_countries]
        data_ = data_[~data_.iso3.isin(exclude_countries)]

    # Create subplots
    proj = ccrs.Robinson(central_longitude=0, globe=None)
    nrows = len(variables) + 1
    ncols = 3
    fig_width = 12 * centimeter
    hist_width = .34
    scatter_plot_width = 3.84
    map_width = 7.82
    hspace_adjust = .2
    fig_height = (len(variables) * scatter_plot_width + hist_width) * centimeter / (1 - hspace_adjust)
    if fig_height > max_fig_height * centimeter:
        print("Warning. The figure height exceeds the maximum height of 24.7 cm.")
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = GridSpec(nrows, ncols, figure=fig, width_ratios=[map_width, scatter_plot_width, hist_width], height_ratios=[hist_width if not log_xaxis else 0] + [scatter_plot_width] * len(variables))
    plt_axs = [[], [], []]
    for i in range(1, nrows):
        plt_axs[i - 1].append(fig.add_subplot(gs[i, 0], projection=proj))
        plt_axs[i - 1].append(fig.add_subplot(gs[i, 1]))
    plt_axs = np.array(plt_axs)

    if not log_xaxis:
        hist_x_ax = fig.add_subplot(gs[0, 1])
    hist_y_axs = [fig.add_subplot(gs[i, 2]) for i in range(1, nrows)]

    # Plot data
    scatter_x_var = 'gdp_pc_pp' if not log_xaxis else 'log_gdp_pc_pp'
    for i, ((m_ax, s_ax), variable) in enumerate(zip(plt_axs, variables)):
        world_[~world_.iso3.isin(data_.iso3)].boundary.plot(ax=m_ax, fc='lightgrey', lw=0, zorder=0, ec='k')
        m_ax.set_extent([-150, 180, -60, 85])

        # Create a truncated version of the colormap that starts at 20% and goes up to 100%
        truncated_cmap = mcolors.LinearSegmentedColormap.from_list(
            'truncated_cmap', plt.get_cmap(cmap[variable])(np.linspace(0.1, 1, 256))
        )

        norm = None
        if bins_list[variable] is not None:
            norm = BoundaryNorm(bins_list[variable], truncated_cmap.N)
        data_.plot(column=variable, ax=m_ax, zorder=5, cmap=truncated_cmap, norm=norm, lw=0, legend=True,
                       legend_kwds={'orientation': 'horizontal', 'shrink': 0.6, 'aspect': 30, 'fraction': .1, 'pad': 0})

        m_ax.axis('off')

        # rasterize m_ax
        m_ax.set_rasterized(True)

        if run_ols:
            add_regression(s_ax, data_, scatter_x_var, variable)

        sns_plot = sns.scatterplot(data=data_, x=scatter_x_var, y=variable, ax=s_ax,
                                   hue='Country income group', hue_order=['LICs', 'LMICs', 'UMICs', 'HICs'], alpha=.5,
                                   palette=INCOME_GROUP_COLORS,
                                   style='Country income group', markers=INCOME_GROUP_MARKERS, s=10,
                                   legend='brief' if i == 1 else False)

        if i == 1:
            sns.move_legend(sns_plot, 'upper right', bbox_to_anchor=(1, 1), frameon=False, title=None, handletextpad=-.25)
            # Adjust legend markers to have full opacity
            for legend_handle in sns_plot.legend_.legend_handles:
                legend_handle.set_alpha(1)

        if annotate:
            for idx, row in data_[data_.iso3.isin(annotate)].iterrows():
                txt = row.iso3
                s_ax.annotate(txt, (row.loc[scatter_x_var], row.loc[variable]), fontsize=6, ha='center', va='top')

        # Histogram for the distribution of 'Country income group'
        sns.kdeplot(data=data_, y=variable, hue='Country income group', ax=hist_y_axs[i], legend=False, fill=False,
                    common_norm=True,
                    palette=INCOME_GROUP_COLORS, alpha=.5)
        hist_y_axs[i].set_ylim(s_ax.get_ylim())

        hist_y_axs[i].axis('off')

        if i == 0 and not log_xaxis:
            # sns.kdeplot(data=data_, x='gdp_pc_pp', ax=hist_x_ax, legend=False, fill=True, common_norm=True, lw=0, alpha=.25, color='k')
            sns.kdeplot(data=data_, x='gdp_pc_pp', hue='Country income group', ax=hist_x_ax, legend=False, fill=False, common_norm=True,
                        palette=INCOME_GROUP_COLORS, alpha=.5)
            hist_x_ax.set_xlim(s_ax.get_xlim())

            hist_x_ax.axis('off')

        s_ax.set_title('')
        s_ax.set_xlabel('')
        s_ax.set_ylabel('')

    for ax in plt_axs[:-1, 1]:
        ax.set_xticklabels([])

    plt_axs[-1, 1].set_xlabel(NAME_DICT[scatter_x_var])

    plt.tight_layout(pad=0)

    # add space between the subplots
    plt.subplots_adjust(hspace=hspace_adjust)

    for i, ((m_ax, s_ax), variable) in enumerate(zip(plt_axs, variables)):
        fig.text(0, .5, NAME_DICT[variable], transform=blended_transform_factory(fig.transFigure, m_ax.transAxes),
                 rotation=90, va='top', ha='center', rotation_mode='anchor')

        # Adjust the position of hist_y_axs to align with s_ax
        pos_s_ax = s_ax.get_position()
        pos_hist_y_ax = hist_y_axs[i].get_position()
        width_new = (pos_hist_y_ax.x1 - pos_s_ax.x1) * .95
        x0_new = pos_s_ax.x1 + (pos_hist_y_ax.x1 - pos_s_ax.x1) * .05
        hist_y_axs[i].set_position([x0_new, pos_hist_y_ax.y0, width_new, pos_hist_y_ax.height])

        if i == 0 and not log_xaxis:
            pos_hist_x_ax = hist_x_ax.get_position()
            height_new = (pos_hist_x_ax.y1 - pos_s_ax.y1) * .95
            y0_new = pos_s_ax.y1 + (pos_hist_x_ax.y1 - pos_s_ax.y1) * .05
            hist_x_ax.set_position([pos_hist_x_ax.x0, y0_new, pos_hist_x_ax.width, height_new])

    if numbering:
        i = 0
        for row in plt_axs:
            bbox = row[1].get_position()
            fig.text(0, bbox.y1, f'{chr(97 + i)}', ha='left', va='top', fontsize=8, fontweight='bold')
            fig.text(bbox.x0 * .85, bbox.y1, f'{chr(97 + i + 1)}', ha='left', va='top', fontsize=8, fontweight='bold')
            i += 2

    # Save or show the plot
    if outfile:
        plt.savefig(outfile, dpi=900, bbox_inches='tight', transparent=True, pad_inches=0)
    if show:
        plt.show(block=False)
        return
    else:
        plt.close()
        return plt_axs


def plot_fig_3(results_data_, cat_info_data_, outfile=None, show=False, numbering=True):
    fig_width = single_col_width * centimeter * 0.8
    fig_heigt = 2 * fig_width
    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(fig_width, fig_heigt),
                            gridspec_kw={'height_ratios': [.75, 4, .75, 4], 'hspace': 0},
                            )
    ax_kde = axs[0]
    ax_scatter = axs[1]
    ax_boxplots = axs[3]
    axs[2].set_visible(False)
    # Scatter plot
    sns_scatter = sns.scatterplot(data=results_data_, x='gini_index', y='resilience', hue='Country income group', style='Country income group',
                                  palette=INCOME_GROUP_COLORS, ax=ax_scatter, legend=True, markers=INCOME_GROUP_MARKERS, alpha=.5, s=10,
                                  hue_order=['LICs', 'LMICs', 'UMICs', 'HICs'])
    sns.move_legend(sns_scatter, 'upper right', bbox_to_anchor=(1, 1), frameon=False, title=None, handletextpad=-.25)
    # Adjust legend markers to have full opacity
    for legend_handle in sns_scatter.legend_.legend_handles:
        legend_handle.set_alpha(1)
    ax_scatter.set_xlabel(NAME_DICT['gini_index'])
    ax_scatter.set_ylabel(NAME_DICT['resilience'])

    add_regression(ax_scatter, results_data_, 'gini_index', 'resilience', p_val_pos='lower left')

    # KDE plot
    sns.kdeplot(data=results_data_, x='gini_index', hue='Country income group', palette=INCOME_GROUP_COLORS, ax=ax_kde, fill=False, common_norm=False, legend=False, alpha=0.5)
    ax_kde.set_xlim(ax_scatter.get_xlim())
    ax_kde.set_xlabel('')
    ax_kde.set_ylabel('')
    ax_kde.set_xticklabels([])
    ax_kde.set_yticklabels([])
    ax_kde.set_xticks([])
    ax_kde.set_yticks([])
    ax_kde.axis('off')

    boxplot_data = cat_info_data_[['c']].droplevel(['hazard', 'rp', 'affected_cat', 'helped_cat']).drop_duplicates()
    for variable in ['dk', 'dw']:
        var_data = cat_info_data_[[variable, 'n']].prod(axis=1).groupby(['iso3', 'hazard', 'rp', 'income_cat']).sum()
        var_data = average_over_rp(var_data).groupby(['iso3', 'income_cat']).sum()
        boxplot_data[variable] = var_data
        boxplot_data[f'{variable}_rel'] = var_data / var_data.groupby('iso3').sum()

    boxplot_data = boxplot_data[['dw_rel', 'dk_rel']].stack() * 100
    boxplot_data.index.names = ['iso3', 'Household income quintile', 'loss_type']
    boxplot_data.name = 'shares [%]'
    boxplot_data = boxplot_data.reset_index()
    boxplot_data = boxplot_data.replace({'dw_rel': 'wellbeing loss', 'dk_rel': 'asset loss'})

    # print relative wellbeing and asset loss share statistics for each Household income quintile
    print('Relative wellbeing and asset loss share statistics for each Household income quintile:')
    print(boxplot_data.groupby(['Household income quintile', 'loss_type'])['shares [%]'].describe())

    sns_boxplot = sns.boxplot(x='Household income quintile', y='shares [%]', hue='loss_type',
                linewidth=.5, legend='brief', ax=ax_boxplots,
                palette=[plt.get_cmap('Blues')(.5), plt.get_cmap('Greens')(.5)],
                data=boxplot_data, flierprops=dict(marker='o', markersize=1, markerfacecolor='black', lw=0, alpha=.5))
    sns.move_legend(sns_boxplot, loc='best', frameon=False, title=None)

    ax_boxplots.set_ylabel('Loss share [%]')

    plt.tight_layout(pad=0, w_pad=1.08)

    if numbering:
        fig.text(-.27, 1, f'{chr(97)}', ha='left', va='top', fontsize=8, fontweight='bold', transform=ax_scatter.transAxes)
        fig.text(-.28, 1, f'{chr(98)}', ha='left', va='top', fontsize=8, fontweight='bold', transform=ax_boxplots.transAxes)

    if outfile:
        plt.savefig(outfile, dpi=300, bbox_inches='tight')
    if show:
        plt.show(block=False)
    else:
        plt.close()


def plot_fig_5(results_data_, cat_info_data_, plot_rp=None, outfile=None, show=True):
    variables = {
        'risk_to_assets': 'Avoided risk to\nassets [%]',
        'risk': 'Avoided risk to\nwellbeing [%]',
        'resilience': 'Socioeconomic resilience\nchange [pp]',
        'recovery increase': 'Recovery time\nreduction [%]',
    }

    policy_scenarios = {
        'reduce_total_exposure_0.05': '1: Reduce total exposure by 5%',
        'reduce_poor_exposure_0.05': '2: Reduce total exposure by 5%\n    targeting the poor',
        'reduce_total_vulnerability_0.05': '3: Reduce total vulnerability by 5%',
        'reduce_poor_vulnerability_0.05': '4: Reduce total vulnerability by 5%\n    targeting the poor',
        'increase_gdp_pc_and_liquidity_0.05': '5: Increase GDP and liquidity by 5%',
        'reduce_gini_10': '6: Equally redistribute 10%\n    of all income',
        'reduce_self_employment_0.1': '7: Reduce self-employment\n    rate by 10%',
        'reduce_non_diversified_income_0.1': '8: Reduce non-diversified\n    income by 10%',
        'pds40': '9: Imperfect PDS aiming at\n    40% of asset losses of the poor',
        'insurance20': '10: National insurance program\n      covering 20% of all asset losses',
    }
    ref_data = results_data_['baseline'].copy()
    # ref_data['t_reco_95_avg'] = cat_info_data_['baseline'].loc[pd.IndexSlice[:, :, plot_rp, :, 'a', :], ['t_reco_95', 'n']].groupby('iso3').apply(lambda x: np.average(x['t_reco_95'], weights=x['n']))
    ref_data['t_reco_95_avg'] = compute_average_recovery_duration(cat_info_data_['baseline'], 'iso3', plot_rp)
    fig_width = double_col_width * centimeter
    fig_height = 2 * centimeter * (len(policy_scenarios) - 1)
    fig, axs = plt.subplots(nrows=1, ncols=len(variables), figsize=(fig_width, fig_height), sharex='col', sharey=True)
    differences = []
    for data_idx, (scenario, scenario_name) in enumerate(policy_scenarios.items()):
        difference = (1 - results_data_[scenario][['risk_to_assets', 'risk']].mul(results_data_[scenario].gdp_pc_pp, axis=0) / ref_data[['risk_to_assets', 'risk']].mul(ref_data.gdp_pc_pp, axis=0)) * 100
        difference['resilience'] = results_data_[scenario]['resilience'] - ref_data['resilience']
        t_reco_95_avg = compute_average_recovery_duration(cat_info_data_[scenario], 'iso3', plot_rp)
        difference['recovery increase'] = (1 - t_reco_95_avg / ref_data.t_reco_95_avg) * 100
        difference['Country income group'] = ref_data['Country income group']
        difference = difference.assign(scenario=scenario_name)
        differences.append(difference)
    differences = pd.concat(differences)

    # compute avoided wellbeing losses over avoided asset losses by country income group
    reduced_assets_impact = differences[differences.scenario.isin(list(policy_scenarios.values())[:6])].copy()
    reduced_assets_impact['avoided_dw_over_avoided_dk'] = reduced_assets_impact['risk'] / reduced_assets_impact['risk_to_assets']
    print(reduced_assets_impact.groupby(['scenario', 'Country income group']).avoided_dw_over_avoided_dk.describe())

    pds40_return = (results_data_['baseline'][['risk', 'gdp_pc_pp']].prod(axis=1) - results_data_['pds40'][['risk', 'gdp_pc_pp']].prod(axis=1)) / 100 / results_data_['pds40']['help_received']
    insurance20_return = (results_data_['baseline'][['risk', 'gdp_pc_pp']].prod(axis=1) - results_data_['insurance20'][['risk', 'gdp_pc_pp']].prod(axis=1)) / 100 / results_data_['insurance20']['help_received']
    returns_merged = pd.merge(
        pd.concat([pds40_return.rename('pds40'), insurance20_return.rename('incurance20')], axis=1),
        income_groups, left_index=True, right_index=True
    )
    print(returns_merged.groupby('Country income group')[['pds40', 'incurance20']].describe())

    for ax, (var, var_name) in zip(axs, variables.items()):
        legend = False
        if var == 'recovery increase':
            legend = True
        sns.boxplot(
            data=differences,
            y='scenario',
            x=var,
            ax=ax,
            orient='h',
            showfliers=False,
            hue='Country income group',
            hue_order=['LICs', 'LMICs', 'UMICs', 'HICs'],
            legend=legend,
            width=.5,
            palette=INCOME_GROUP_COLORS,
            fill=False,
            gap=0.35,
            linewidth=1,
        )
        ax.axvline(0, color='k', lw=1)

        # Plot the vertical grid lines
        ax.grid(axis='x', linestyle='--', color='grey', alpha=0.7)

        # Hide the y-ticks but keep the y-tick labels
        ax.tick_params(axis='y', length=0)

        # Remove all spines except the bottom one
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        ax.set_xlabel(var_name)

        ax.set_facecolor('whitesmoke')

    for label in axs[0].get_yticklabels():
        label.set_horizontalalignment('left')
    axs[0].tick_params(axis='y', pad=110)

    axs[0].set_ylabel('')
    axs[-1].legend(title=None, frameon=False, bbox_to_anchor=(1, 1), loc='upper left')

    for ax_idx, ax in enumerate(axs):
        ax.text(0.5, 1.01, f'{chr(97 + ax_idx)}', ha='center', va='bottom', fontsize=8, fontweight='bold', transform=ax.transAxes)

    plt.tight_layout(pad=0, w_pad=2, h_pad=2)

    if show:
        plt.show(block=False)
    if outfile:
        plt.savefig(outfile, dpi=300, bbox_inches='tight')


def plot_supfig_7(results_data_, cat_info_data_, plot_rp=None, outfile=None, show=False):
    fig_width = single_col_width * centimeter
    fig_heigt = fig_width
    fig, ax = plt.subplots(figsize=(fig_width, fig_heigt))

    duration_ctry = pd.merge(
        compute_average_recovery_duration(cat_info_data_, 'iso3', plot_rp),
        income_groups['Country income group'],
        left_index=True,
        right_index=True
    )
    plot_data = pd.merge(results_data_[['resilience', 'Country income group']], duration_ctry.t_reco_avg, left_index=True, right_index=True)

    # Scatter plot
    sns_scatter = sns.scatterplot(
        data=plot_data,
        x='resilience', y='t_reco_avg', hue='Country income group', style='Country income group',
        palette=INCOME_GROUP_COLORS, ax=ax, legend=True, markers=INCOME_GROUP_MARKERS, alpha=.5, s=10,
        hue_order=['LICs', 'LMICs', 'UMICs', 'HICs']
    )
    ax.set_xlabel('Socio-economic resilience [%]')
    ax.set_ylabel('Average recovery\nduration [yr]')

    add_regression(ax, plot_data, 'resilience', 't_reco_avg', p_val_pos='upper left')

    sns.move_legend(sns_scatter, 'best', frameon=False, title=None, handletextpad=-.25)

    plt.tight_layout(pad=0.2, w_pad=1.08)

    if outfile:
        plt.savefig(outfile, dpi=300, bbox_inches='tight')
    if show:
        plt.show(block=False)
    else:
        plt.close()


def plot_hazard_detail_supfig(cat_info_data_, income_groups_, plot_rp=100, outfile=None, show=False):
    hazards = cat_info_data_.index.get_level_values('hazard').unique()
    fig_width = double_col_width * centimeter
    fig_height = fig_width * .5
    fig, axs = plt.subplots(ncols=len(hazards), nrows=2, figsize=(fig_width, fig_height), sharex='col', sharey='row')

    plot_data = pd.merge(cat_info_data_['t_reco_95'], income_groups_['Country income group'], left_index=True, right_index=True, how='inner')

    for idx, (hazard, axs_col) in enumerate(zip(hazards, axs.T)):
        legend=False
        if idx == len(hazards) - 1:
            legend = True
        helped_slice = pd.IndexSlice[:, hazard, plot_rp, :, 'a', 'helped']
        not_helped_slice = pd.IndexSlice[:, hazard, plot_rp, :, 'a', 'not_helped']
        errorbar = lambda x: tuple(np.percentile(x, [25, 75]))
        sns.barplot(plot_data.loc[not_helped_slice], x='Country income group', y='t_reco_95', hue='income_cat', order=['LICs', 'LMICs', 'UMICs', 'HICs'], ax=axs_col[0], err_kws={'linewidth': 0.75}, legend=False, estimator='median', errorbar=errorbar)
        sns.barplot(plot_data.loc[helped_slice], x='Country income group', y='t_reco_95', hue='income_cat', order=['LICs', 'LMICs', 'UMICs', 'HICs'], ax=axs_col[1], err_kws={'linewidth': 0.75}, legend=legend, estimator='median', errorbar=errorbar)

        axs_col[0].set_title(hazard)

        axs_col[1].set_xlabel('')
        axs_col[1].set_ylabel('')

        for ig in ['LICs', 'LMICs', 'UMICs', 'HICs']:
            axs_col[0].text(ig, .95, f"({int(plot_data.loc[not_helped_slice].groupby('Country income group')['t_reco_95'].count().loc[ig] / 5)})", ha='center', va='top',
                            transform=blended_transform_factory(axs_col[0].transData, axs_col[0].transAxes), fontsize=7)

    axs[-1, -1].legend(title=None, frameon=False)#bbox_to_anchor=(1, 1), loc='upper left', frameon=False, title='Income\nquintile')

    axs[1, 2].set_xlabel('Country income group')
    axs[0, 0].set_ylabel('Recovery duration of\nnot-helped households [yr]')
    axs[1, 0].set_ylabel('Recovery duration of\nhelped households [yr]')

    plt.tight_layout(pad=.2, w_pad=1.01, h_pad=.5)

    for ax_idx, ax in enumerate(axs.flatten()):
        ax.text(-.08, 1, f'{chr(97 + ax_idx)}', ha='left', va='top', fontsize=8, fontweight='bold', transform=ax.transAxes)

    if outfile:
        plt.savefig(outfile, dpi=300, bbox_inches='tight')
    if show:
        plt.show(block=False)
    else:
        plt.close()


def plot_supfig_8(cat_info_data_, outfile=None, show=False, numbering=True, plot_rp=None):
    fig_width = single_col_width * centimeter
    fig_heigt = .85 * fig_width
    fig, axs = plt.subplots(figsize=(fig_width, fig_heigt), nrows=1, sharex=True)
    axs = [axs]

    t_reco_no_funds = pd.merge(
        compute_average_recovery_duration(cat_info_data_, ['iso3', 'income_cat'], plot_rp),
        income_groups['Country income group'],
        left_index=True,
        right_index=True,
    )
    sns.barplot(
        data=t_reco_no_funds,
        x='income_cat',
        y='t_reco_avg',
        hue='Country income group',
        hue_order=['LICs', 'LMICs', 'UMICs', 'HICs'],
        errorbar=None,
        estimator='median',
        legend=True,
        ax=axs[0],
    )
    # plot median country recovery duration for each Country income group
    for x_val, mean_val in enumerate(t_reco_no_funds.groupby('income_cat').t_reco_avg.median().values.flatten()):
        axs[0].plot([x_val - .4, x_val + .4], [mean_val, mean_val], color='black', lw=1, alpha=.4)

    axs[0].legend(title=None, frameon=False, bbox_to_anchor=(1, 1), loc='upper left')
    axs[0].set_ylabel('Recovery duration [yr]')

    axs[-1].set_xlabel('Household income quintile')

    plt.tight_layout(pad=.2, w_pad=1.08, h_pad=1.08)

    if numbering:
        fig.text(-.19, 1, 'a', ha='left', va='top', fontsize=8, fontweight='bold', transform=axs[0].transAxes)
        fig.text(-.19, 1, 'b', ha='left', va='top', fontsize=8, fontweight='bold', transform=axs[1].transAxes)

    if outfile:
        plt.savefig(outfile, dpi=300, bbox_inches='tight')
    if show:
        plt.show(block=False)


def plot_supfig_5(results_data_, outfile=None, show=False):
    fig_width = 14.5 * centimeter
    fig_heigt = 7 * centimeter
    fig, axs = plt.subplots(ncols=2, figsize=(fig_width, fig_heigt))

    # Scatter plot
    sns.scatterplot(data=results_data_, x='risk_to_assets', y='risk', hue='Country income group', style='Country income group',
                                  palette=INCOME_GROUP_COLORS, ax=axs[0], legend=False, markers=INCOME_GROUP_MARKERS, alpha=.5, s=10,
                                  hue_order=['LICs', 'LMICs', 'UMICs', 'HICs'])
    axs[0].set_xlabel(NAME_DICT['risk_to_assets'])
    axs[0].set_ylabel(NAME_DICT['risk'])

    # Scatter plot
    sns_scatter = sns.scatterplot(data=results_data_, x='dk_tot', y='dWtot_currency', hue='Country income group',
                                  style='Country income group',
                                  palette=INCOME_GROUP_COLORS, ax=axs[1], legend=True, markers=INCOME_GROUP_MARKERS,
                                  alpha=.5, s=10,
                                  hue_order=['LICs', 'LMICs', 'UMICs', 'HICs'])
    sns.move_legend(sns_scatter, 'best', frameon=False, title=None, handletextpad=-.25)

    # Adjust legend markers to have full opacity
    for legend_handle in sns_scatter.legend_.legend_handles:
        legend_handle.set_alpha(1)
    dk_tot_label = NAME_DICT['dk_tot'].replace('\n', ' ')
    dw_tot_currency_label = NAME_DICT['dWtot_currency'].replace('\n', ' ')
    axs[1].set_xlabel(f"ln({dk_tot_label})")
    axs[1].set_ylabel(f"ln({dw_tot_currency_label})")

    axs[1].set_yscale('log')
    axs[1].set_xscale('log')

    plt.tight_layout(pad=0, w_pad=1.08)

    fig.text(0, 1, f'{chr(97)}', ha='left', va='top', fontsize=8, fontweight='bold',
             transform=blended_transform_factory(fig.transFigure, axs[0].transAxes))
    fig.text(-.2, 1, f'{chr(98)}', ha='left', va='top', fontsize=8, fontweight='bold',
             transform=axs[1].transAxes)

    if outfile:
        plt.savefig(outfile, dpi=300, bbox_inches='tight')
    if show:
        plt.show(block=False)
    else:
        plt.close()


def plot_supfig_6(results_data_, outfile=None, show=False):
    fig, ax = plt.subplots(figsize=(single_col_width * centimeter, .85 * single_col_width * centimeter))
    sns.scatterplot(
        data=results_data_,
        x='risk_to_assets',
        y='resilience',
        hue='Country income group',
        style='Country income group',
        palette=INCOME_GROUP_COLORS,
        ax=ax,
        markers=INCOME_GROUP_MARKERS,
        alpha=.5,
        s=10,
        hue_order=['LICs', 'LMICs', 'UMICs', 'HICs']
    )
    ax.legend(title=None, frameon=False, bbox_to_anchor=(1, 1), loc='upper left')
    add_regression(
        ax_=ax,
        data_=results_data_,
        x_var_='risk_to_assets',
        y_var_='resilience',
    )
    ax.set_xlabel(NAME_DICT['risk_to_assets'])
    ax.set_ylabel(NAME_DICT['resilience'])
    plt.tight_layout(pad=.1)
    if outfile:
        plt.savefig(outfile, dpi=300, bbox_inches='tight')
    if show:
        plt.show(block=False)
    else:
        plt.close()

def compute_average_recovery_duration(df, aggregation_level, agg_rp=None):
    df = df[['n', 't_reco_95']].xs('a', level='affected_cat').copy()
    df = df[df.n > 0]

    if type(aggregation_level) is str:
        aggregation_level = [aggregation_level]

    if agg_rp is not None:
        return df.xs(agg_rp, level='rp').groupby(aggregation_level).apply(lambda x: np.average(x.t_reco_95, weights=x.n)).rename('t_reco_avg')

    aggregation_level = aggregation_level + ['rp']

    # compute population-weighted average recovery duration of affected households
    df = df.groupby(aggregation_level).apply(lambda x: np.average(x.t_reco_95, weights=x.n)).rename('t_reco_avg')

    if 1 not in df.index.get_level_values('rp'):
        # for aggregation, assume that all events with rp < 10 have the same recovery duration as the event with rp = 10
        rp_1 = df.xs(df.index.get_level_values('rp').min(), level='rp').copy().to_frame()
        rp_1['rp'] = 1
        rp_1 = rp_1.set_index('rp', append=True).reorder_levels(df.index.names).squeeze()
        df = pd.concat([df, rp_1]).sort_index()

    # compute probability of each return period
    return_periods = df.index.get_level_values('rp').unique()

    rp_probabilities = pd.Series(1 / return_periods - np.append(1 / return_periods, 0)[1:], index=return_periods)
    # match return periods and their frequency
    probabilities = pd.Series(data=df.reset_index('rp').rp.replace(rp_probabilities).values, index=df.index,
                              name='probability')

    # average weighted by probability
    res = df.mul(probabilities, axis=0).reset_index('rp', drop=True)
    res = res.groupby(level=list(range(res.index.nlevels))).sum()
    if type(df) is pd.Series:
        res.name = df.name
    return res


def plot_fig_4(cat_info_data_, income_groups_, map_bins, world_, plot_rp, outfile=None, show=False, numbering=True):
    fig_width = single_col_width * centimeter
    fig_height = 2 * fig_width
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = GridSpec(4, ncols=1, figure=fig, height_ratios=[.6, .02, 1, 1])
    axs = []
    map_proj = ccrs.Robinson(central_longitude=0, globe=None)
    axs.append(fig.add_subplot(gs[0], projection=map_proj))
    for i in range(1, 4):
        axs.append(fig.add_subplot(gs[i]))

    # Create a truncated version of the colormap that starts at 20% and goes up to 100%
    truncated_cmap = mcolors.LinearSegmentedColormap.from_list(
        'truncated_cmap', plt.get_cmap('Reds')(np.linspace(0.1, 1, 256))
    )

    # plot median country recovery duration map
    duration_ctry = pd.merge(
        compute_average_recovery_duration(cat_info_data_, 'iso3', plot_rp),
        income_groups['Country income group'],
        left_index=True,
        right_index=True,
    )
    print("Longest recovery durations")
    print(duration_ctry.sort_values(by='t_reco_avg', ascending=False).head(10))

    if map_bins[-1] < duration_ctry.t_reco_avg.max():
        map_bins = np.append(map_bins, np.ceil(duration_ctry.t_reco_avg.max()))
    norm = BoundaryNorm(map_bins, truncated_cmap.N)

    duration_ctry = gpd.GeoDataFrame(pd.merge(duration_ctry, world_, left_index=True, right_on='iso3', how='inner'))
    duration_ctry.plot(column='t_reco_avg', ax=axs[0], cax=axs[1], zorder=5, cmap=truncated_cmap, norm=norm, lw=0, legend=True, legend_kwds={'orientation': 'horizontal', 'shrink': 0.6, 'aspect': 30, 'fraction': .1, 'pad': 0})

    world_[~world_.iso3.isin(duration_ctry.iso3)].boundary.plot(ax=axs[0], fc='lightgrey', lw=0, zorder=0, ec='k')
    axs[0].set_extent([-150, 180, -60, 85])
    axs[0].axis('off')

    axs[0].set_rasterized(True)

    # plot median country recovery duration by hazard for each Country income group
    duration_ctry_hazard = pd.merge(
        compute_average_recovery_duration(cat_info_data_, ['iso3', 'hazard'], plot_rp),
        income_groups_['Country income group'],
        left_index=True,
        right_index=True,
        how='inner'
    )
    duration_ctry_hazard.groupby(['Country income group', 'hazard']).describe()

    sns.barplot(
        data=duration_ctry_hazard.reset_index(),
        x='Country income group',
        y='t_reco_avg',
        hue='hazard',
        ax=axs[2],
        errorbar=None,
        order=['LICs', 'LMICs', 'UMICs', 'HICs'],
        estimator='median',
        palette=HAZARD_COLORS,
    )

    # plot median country recovery duration for each Country income group
    for x_val, mean_val in enumerate(duration_ctry.groupby('Country income group').t_reco_avg.median().loc[['LICs', 'LMICs', 'UMICs', 'HICs']].values.flatten()):
        axs[2].plot([x_val - .4, x_val + .4], [mean_val, mean_val], color='black', lw=1)
        print(f"Median recovery duration for {['LICs', 'LMICs', 'UMICs', 'HICs'][x_val]}: {mean_val}")

    axs[2].set_ylabel('Median recovery duration\nby country income group [yr]')
    axs[2].set_xlabel('Country income group')
    axs[2].legend(frameon=False, title=None, bbox_to_anchor=(1, 1), loc='upper left')

    quintile_data_total = pd.merge(
        # compute_average_recovery_duration(cat_info_data_, ['iso3', 'income_cat', 'hazard'], plot_rp),
        compute_average_recovery_duration(cat_info_data_, ['iso3', 'income_cat'], plot_rp),
        income_groups_['Country income group'],
        left_index=True,
        right_index=True,
    )

    sns.barplot(
        data=quintile_data_total,
        x='income_cat',
        y='t_reco_avg',
        # hue='hazard',
        hue='Country income group',
        hue_order=['LICs', 'LMICs', 'UMICs', 'HICs'],
        errorbar=None,
        estimator='median',
        alpha=1,
        linewidth=.5,
        edgecolor='none',
        legend=True,
        ax=axs[3],
    )
    # plot median recovery duration for each income quintile
    for x_val, mean_val in enumerate(quintile_data_total.groupby('income_cat').t_reco_avg.median().values.flatten()):
        axs[3].plot([x_val - .4, x_val + .4], [mean_val, mean_val], color='black', lw=1)

    axs[2].legend(title=None, frameon=False, bbox_to_anchor=(1, 1), loc='upper left')
    axs[3].legend(title=None, frameon=False, bbox_to_anchor=(1, 1), loc='upper left')
    axs[3].set_ylabel('Median recovery duration\nby household income quintile [yr]')
    axs[3].set_xlabel('Household income quintile')

    plt.tight_layout(pad=.2, w_pad=0, h_pad=1.08)

    # Adjust the position of the map
    m_y0 = axs[2].get_position().y1 + .05
    axs[0].set_position([.05, m_y0, .9, 1 - m_y0])

    # Map axis label
    fig.text(0, 0.5, 'Average recovery\nduration [yr]', fontsize=7, transform=blended_transform_factory(fig.transFigure, axs[0].transAxes), rotation=90, va='top', ha='center', rotation_mode='anchor')

    # Adjust the position of the colorbar
    cbar_y0 = axs[1].get_position().y0
    cbar_y1 = axs[1].get_position().y1
    axs[1].set_position([.25, cbar_y0, .5, cbar_y1 - cbar_y0])

    if numbering:
        for idx, ax in enumerate([axs[0], axs[2], axs[3]]):
            fig.text(0, 1, f'{chr(97 + idx)}', ha='left', va='top', fontsize=8, fontweight='bold', transform=blended_transform_factory(fig.transFigure, ax.transAxes))

    if outfile:
        plt.savefig(outfile, dpi=900, bbox_inches='tight')
    if show:
        plt.show(block=False)

    return fig, axs


def plot_recovery_comparison(macro_baseline, cat_info_baseline, macro_no_liquidity, cat_info_no_liquidity, household_, outfile=None):
    fig_width = double_col_width * centimeter
    fig_heigt = 10 * centimeter
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(fig_width, fig_heigt), sharex=True, sharey='row')

    plot_data_no_liquidity = pd.merge(macro_no_liquidity, cat_info_no_liquidity, left_index=True, right_index=True).loc[tuple(household_ + ['not_helped'])]
    plot_recovery(6, plot_data_no_liquidity.avg_prod_k, plot_data_no_liquidity.tau_tax, plot_data_no_liquidity.k, plot_data_no_liquidity.dk,
                  plot_data_no_liquidity.lambda_h, plot_data_no_liquidity.k_household_share, plot_data_no_liquidity.liquidity, plot_data_no_liquidity.help_received, np.nan,
                  plot_data_no_liquidity.recovery_params, plot_data_no_liquidity.gamma_SP * plot_data_no_liquidity.n, plot_data_no_liquidity.diversified_share,
                  ylims=[(-1000, 5500), None],
                  title=f"without liquidity", fig=fig, axs=axs[:, 0], plot_legend=False)

    data_baseline = pd.merge(macro_baseline, cat_info_baseline, left_index=True, right_index=True).loc[tuple(household_ + ['helped'])]
    plot_recovery(6, data_baseline.avg_prod_k, data_baseline.tau_tax, data_baseline.k, data_baseline.dk,
                  data_baseline.lambda_h, data_baseline.k_household_share, data_baseline.liquidity, data_baseline.help_received, np.nan,
                  data_baseline.recovery_params, data_baseline.gamma_SP * data_baseline.n, data_baseline.diversified_share,
                  ylims=[(-1000, 5500), None],
                  title=f"with liquidity", fig=fig, axs=axs[:, 1], show_ylabel=False)
    for ax in axs.flatten():
        ax.set_yticks([0])
        ax.set_yticklabels([0])
        ax.set_xticks([0])
        ax.set_xticklabels([0])
    for ax in axs[0, :]:
        ax.axhline(0, color='black', lw=1, linestyle='--')
    for ax in axs[1, :]:
        ax.set_xlabel('Time')
    axs[0, 0].set_ylabel('Consumption')
    axs[1, 0].set_ylabel('Capital loss')
    plt.tight_layout()
    if outfile:
        plt.savefig(
            outfile, dpi=300, bbox_inches='tight')


def print_stats(results_data_):
    # print the asset losses, wellbeing losses, and resilience for TJK and HTI
    for c in ['HTI', 'TJK']:
        print(f'{c}: dK={results_data_.loc[c, "dk_tot"] / 1e9}, dC^eq={results_data_.loc[c, "dWtot_currency"] / 1e9}, Psi={results_data_.loc[c, "resilience"]}')

    # print the population-weighted average resilience
    print(f'Global: dK={results_data_.dk_tot.sum() / 1e9}, dC^eq={results_data_.dWtot_currency.sum() / 1e9}, resilience={results_data_.dk_tot.sum() / results_data_.dWtot_currency.sum()}')

    # print the stats of the resilience by Country income group
    print('Resilience by Country income group:')
    print(results_data_.groupby('Country income group').resilience.describe().loc[['LICs', 'LMICs', 'UMICs', 'HICs']])

    # print minimum and maximum resilience of each Country income group
    print('Minimum and maximum resilience of each Country income group:')
    print(results_data_.groupby('Country income group').resilience.agg(['min', 'max', 'idxmin', 'idxmax']))


def print_results_table(results_data_):
    for idx, row in results_data_.iterrows():
        print(f'{idx}  & {row["Country income group"]} & {row["gdp_pc_pp"]:.0f} & {row["risk_to_assets"]:.2f} & {row["risk"]:.2f}  & {row["resilience"]:.2f}\\\\')


def plot_scatter(data, x_vars, y_vars, exclude_countries=None, reg_degrees=None, name_dict=None, outfile=None,
                 xlim=None, ylim=None, plot_unit_line=False, annotate=None):
    if isinstance(data, str):
        data = pd.read_csv(data)
    elif not isinstance(data, pd.DataFrame):
        raise ValueError('data should be a path to a csv file or a pandas DataFrame.')
    if exclude_countries:
        if isinstance(exclude_countries, str):
            exclude_countries = [exclude_countries]
        data = data[~data.iso3.isin(exclude_countries)]

    if isinstance(x_vars, str):
        x_vars = [x_vars]
    if isinstance(y_vars, str):
        y_vars = [y_vars]

    if len(x_vars) != len(y_vars):
        raise ValueError('The number of y_vars and x_vars should be the same')

    if isinstance(reg_degrees, int):
        reg_degrees = [reg_degrees] * len(x_vars)
    elif reg_degrees is not None and len(reg_degrees) != len(x_vars):
        raise ValueError('The number of reg_degree should be the same as the number of x_vars')
    elif reg_degrees is None:
        reg_degrees = [None] * len(x_vars)

    if not (isinstance(xlim, dict) or isinstance(xlim, tuple) or xlim is None):
        raise ValueError('xlim should be a dictionary, tuple, or None.')
    if isinstance(xlim, tuple):
        xlim = {xl: xlim for xl in x_vars}
    elif xlim is None:
        xlim = {}
    if not (isinstance(ylim, dict) or isinstance(ylim, tuple) or ylim is None):
        raise ValueError('ylim should be a dictionary, tuple, or None.')
    if isinstance(ylim, tuple):
        ylim = {yl: ylim for yl in y_vars}
    elif ylim is None:
        ylim = {}

    fig, axs = plt.subplots(1, len(x_vars), figsize=(6 * len(x_vars), 5))
    if isinstance(axs, plt.Axes):
        axs = [axs]

    for x_var, y_var, ax, reg_degree in zip(x_vars, y_vars, axs, reg_degrees):
        data_ = data.dropna(subset=[x_var, y_var])
        ax.scatter(data_[x_var], data_[y_var], alpha=.65, lw=0)
        if annotate:
            if annotate == 'all':
                for i, row in data_[['iso3']].iterrows():
                    txt = row.iso3
                    ax.annotate(txt, (data_[x_var].loc[i], data_[y_var].loc[i]))
            else:
                for i, row in data_[data_.iso3.isin(annotate)][['iso3']].iterrows():
                    txt = row.iso3
                    ax.annotate(txt, (data_.loc[i, x_var], data_.loc[i, y_var]))

        if reg_degree:
            if isinstance(reg_degree, str):
                reg_degree = reg_degrees[reg_degree]
            if isinstance(reg_degree, int) or reg_degree is None:
                reg_degree = [reg_degree]

            for d_idx, (degree, color) in enumerate(zip(reg_degree, ['red', 'blue', 'green', 'orange', 'purple'])):
                # Fit a polynomial to the data
                p = polyfit(data_[x_var], data_[y_var], degree)
                x_line = np.linspace(min(data_[x_var]), max(data_[x_var]), 100)
                y_line = polyval(x_line, p)
                ax.plot(x_line, y_line, color=color)

                # Calculate the R-squared value
                y_pred = polyval(data_[x_var], p)
                r_squared = r2_score(data_[y_var], y_pred)
                ax.text(0.1, 0.9 - .05 * d_idx, r'$R^2 = $ {:.3f}'.format(r_squared), transform=ax.transAxes,
                        color=color)

        format_axis(ax, x_name=x_var, y_name=y_var, name_mapping=name_dict, xlim=xlim.get(x_var), ylim=ylim.get(y_var))

        if plot_unit_line:
            ax.axline([0, 0], slope=1, ls='--', c='k', lw=.5, label='y=x')
            ax.legend()

    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.show(block=False)



def plot_hbar(data, variables, comparison_data=None, norm=None, how='abs', head=15, outfile=None,
              name_dict=None, unit=None, precision=None):
    if isinstance(variables, str):
        variables = [variables]
    if precision is None:
        precision = 2
    # data = pd.read_csv(data_path)
    if comparison_data is not None:
        # comparison_data = pd.read_csv(comparison_data_path)
        data = pd.merge(data, comparison_data, on='iso3', suffixes=('', '_comp'))
        for variable in variables:
            if how == 'abs':
                data[variable] = data[variable] - data[variable + '_comp']
                xlabel = 'absolute difference'
            elif how == 'rel':
                data[variable] = ((data[variable] - data[variable + '_comp']) / data[variable]) * 100
                xlabel = 'Difference (%)'
            else:
                raise ValueError(f"unknown value '{how}' for parameter 'how'. Use 'abs' or 'rel'.")
    else:
        xlabel = 'Value'

    if norm is not None:
        if norm == 'GDP':
            data[variables] = data[variables].div(data[['gdp_pc_pp', 'pop']].prod(axis=1), axis=0) * 100
            xlabel = f'{xlabel} (% GDP)'
        else:
            data[variables] = data[variables].div(data[norm], axis=0) * 100
            xlabel = f'{xlabel} (% {norm})'

    data['country'] = data.iso3.map(iso3_to_wb)
    data = data[variables + ['country']].sort_values(by=variables[0], ascending=True).tail(head)

    if unit == 'millions':
        data[variables] = data[variables] / 1e6
        xlabel = f'{xlabel} (m $PPP)'
    elif unit == 'billions':
        data[variables] = data[variables] / 1e9
        xlabel = f'{xlabel} (bn $PPP)'

    fig, ax = plt.subplots(1, 1, figsize=(4, 6))
    bars = data[variables[::-1]].rename(columns=name_dict).plot(kind='barh', ax=ax, legend=False, width=.8,
                                                                color=[plt.get_cmap('Greens')(0.5), plt.get_cmap('Blues')(0.5)])

    # Add the value behind each bar
    for bar in bars.containers:
        ax.bar_label(bar, fmt=f'%.{precision}f', label_type='edge')

    plt.legend(loc='lower left', bbox_to_anchor=(0, 1.01), frameon=False)
    ax.set_yticklabels(data.country)
    ax.set_xlabel(xlabel)

    # Disable top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.show(block=False)


def fancy_scatter_matrix(data, reg_degree=1):
    if isinstance(reg_degree, int):
        reg_degree = [reg_degree]
    axes = pd.plotting.scatter_matrix(data, diagonal='kde')
    # corr = data.corr().to_numpy()

    # for i, j in zip(*plt.np.tril_indices_from(axes, k=-1)):
    for i, j in itertools.product(np.arange(axes.shape[0]), np.arange(axes.shape[1])):
        if i == j:
            continue
        i_name = data.columns[i]
        j_name = data.columns[j]
        x_data_j = data.dropna(subset=[i_name, j_name])[j_name]
        y_data_i = data.dropna(subset=[i_name, j_name])[i_name]
        ax = axes[i, j]
        # ax.annotate(f"r={np.round(corr[i, j], 3)}", (.05, .1), xycoords='axes fraction', ha='left',
        #                     va='center', color='r', size=8)
        for d_idx, (degree, color) in enumerate(zip(reg_degree, ['red', 'blue', 'green', 'orange', 'purple'])):
            # Fit a polynomial to the data
            p = polyfit(x_data_j, y_data_i, degree)
            x_line = np.linspace(min(x_data_j), max(x_data_j), 100)
            y_line = polyval(x_line, p)
            ax.plot(x_line, y_line, color=color, lw=.5, alpha=.75)
            # Calculate the R-squared value
            y_pred = polyval(x_data_j, p)
            r_squared = r2_score(y_data_i, y_pred)
            ax.text(0.01, 0.99 - .15 * d_idx, r'$R^2 = $ {:.3f}'.format(r_squared), transform=ax.transAxes,
                    color=color, ha='left', va='top', alpha=.75)
    plt.show(block=False)


def plot_fast_slow_recovery_example_figure():
    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(6, 4), sharex=True, sharey='row', height_ratios=[.5, .8])
    c_baseline = 30000
    pi = .3
    k_baseline = c_baseline / pi
    dk_0 = 0.25 * k_baseline
    lambda_slow = 0.25
    lambda_fast = 1
    t = np.linspace(0, 10, 5000)
    for col, l in enumerate([lambda_fast, lambda_slow]):
        dk = dk_0 * np.exp(-l * t)
        dc_lab = dk * pi
        dc_reco = l * dk
        dc = dc_lab + dc_reco
        axs[0, col].plot([-.25, 0, 0], [k_baseline, k_baseline, k_baseline - dk_0], label='__none__', color='k')
        axs[1, col].plot([-.25, 0, 0], [c_baseline, c_baseline, c_baseline - dc[0]], label='__none__', color='k')
        axs[0, col].plot(t, k_baseline - dk, label='__none__', color='k')
        axs[1, col].fill_between(t, c_baseline, c_baseline - dc_lab, color='red', alpha=.5,
                                 lw=0, label=r'Income loss $\Delta c^{lab}$')
        axs[1, col].fill_between(t, c_baseline - dc_lab, c_baseline - dc_lab - dc_reco, color='red', alpha=.25, lw=0,
                                 label=r'Reconstruction loss $\Delta c^{reco}$',)
        axs[1, col].plot(t, c_baseline - dc_lab - dc_reco, color='k', label='__none__')
    axs[0, 0].set_ylabel('Capital stock')
    axs[1, 0].set_ylabel('Consumption')
    axs[1, 0].set_xlabel('Time after disaster')
    axs[1, 1].set_xlabel('Time after disaster')
    for ax in axs.flatten():
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticks([0, 2.5, 5, 7.5, 10])
        ax.set_xticklabels([r'$t_0$', None, None, None, None])
        ax.set_yticklabels([])
    axs[-1, -1].legend(loc='lower right', frameon=False)
    axs[0, 0].set_title('Fast recovery')
    axs[0, 1].set_title('Slow recovery')
    plt.tight_layout()


# def load_income_groups():
#     income_groups_ = pd.read_excel(os.path.join(input_data_dir, 'WB_country_classification/country_classification.xlsx'), header=0)[["Code", "Region", "Income group"]]
#     income_groups_ = income_groups_.dropna().rename({'Code': 'iso3'}, axis=1)
#     income_groups_ = income_groups_.set_index('iso3').squeeze()
#     income_groups_.loc['VEN'] = ['Latin America & Caribbean', 'Upper middle income']
#     income_groups_.rename({'Income group': 'Country income group'}, axis=1, inplace=True)
#     income_groups_.replace({'Low income': 'LICs', 'Lower middle income': 'LMICs', 'Upper middle income': 'UMICs', 'High income': 'HICs'}, inplace=True)
#     return income_groups_


def plot_ew_impact_over_time(ensemble_dir, outpath_=None):
    upper_bound_results = None
    lower_bound_results = None
    for ew_year in np.arange(1987, 2019):
        upper_bound_dir = [d for d in os.listdir(ensemble_dir) if d.endswith(f"baseline_EW-{ew_year}")]
        lower_bound_dir = [d for d in os.listdir(ensemble_dir) if "_reduce_ew_" in d and f"{ew_year}" in d]
        if len(upper_bound_dir) != 1 or len(lower_bound_dir) != 1:
            raise ValueError(f"Expected exactly one directory for upper and lower bound simulations, found {len(upper_bound_dir)} and {len(lower_bound_dir)}")
        upper_bound_dir = upper_bound_dir[0]
        lower_bound_dir = lower_bound_dir[0]
        upper_bound_path = os.path.join(ensemble_dir, upper_bound_dir, 'results_tax_unif_poor.csv')
        lower_bound_path = os.path.join(ensemble_dir, lower_bound_dir, 'results_tax_unif_poor.csv')
        upper_bound_year_results = pd.read_csv(upper_bound_path, index_col=0)[['dk_tot', 'dWtot_currency']].rename(columns={'dk_tot': 'Asset losses', 'dWtot_currency': 'Welfare losses'}).stack().rename(ew_year)
        lower_bound_year_results = pd.read_csv(lower_bound_path, index_col=0)[['dk_tot', 'dWtot_currency']].rename(columns={'dk_tot': 'Asset losses', 'dWtot_currency': 'Welfare losses'}).stack().rename(ew_year)
        if upper_bound_results is None:
            upper_bound_results = upper_bound_year_results
        else:
            upper_bound_results = pd.concat([upper_bound_results, upper_bound_year_results], axis=1)
        if lower_bound_results is None:
            lower_bound_results = lower_bound_year_results
        else:
            lower_bound_results = pd.concat([lower_bound_results, lower_bound_year_results], axis=1)
    no_ew_dir = [d for d in os.listdir(ensemble_dir) if f"set_ew_0.0_EW-2018" in d]
    full_ew_dir = [d for d in os.listdir(ensemble_dir) if f"set_ew_1.0_EW-2018" in d]
    if len(no_ew_dir) != 1:
        raise ValueError(f"Expected exactly one directory containing 'set_ew_0.0_EW-2018', found {len(no_ew_dir)}")
    no_ew_results = pd.read_csv(os.path.join(ensemble_dir, no_ew_dir[0], 'results_tax_unif_poor.csv'), index_col=0)[['dk_tot', 'dWtot_currency']].rename({'dk_tot': 'Asset losses', 'dWtot_currency': 'Welfare losses'}).stack().rename('no_ew')
    if len(full_ew_dir) != 1:
        raise ValueError(f"Expected exactly one directory containing 'set_ew_1.0_EW-2018', found {len(full_ew_dir)}")
    full_ew_results = pd.read_csv(os.path.join(ensemble_dir, full_ew_dir[0], 'results_tax_unif_poor.csv'), index_col=0)[['dk_tot', 'dWtot_currency']].rename({'dk_tot': 'Asset losses', 'dWtot_currency': 'Welfare losses'}).stack().rename('full_ew')
    upper_bound_avoided_losses = pd.DataFrame(data=no_ew_results.to_frame().values - upper_bound_results.values, columns=upper_bound_results.columns,
                                  index=upper_bound_results.index)
    lower_bound_avoided_losses = pd.DataFrame(data=no_ew_results.to_frame().values - lower_bound_results.values, columns=lower_bound_results.columns,
                                    index=lower_bound_results.index)
    mean_avoided_losses = (upper_bound_avoided_losses + lower_bound_avoided_losses) / 2
    diff_avoided_losses = (upper_bound_avoided_losses - lower_bound_avoided_losses) / 2
    upper_bound_possible_avoided_losses = pd.Series(data=no_ew_results.values - full_ew_results.values, index=upper_bound_results.index,
                         name='upper_bound_possible_avoided_losses')
    lower_bound_possible_avoided_losses = pd.Series(data=no_ew_results.values - full_ew_results.values, index=lower_bound_results.index,
                            name='lower_bound_possible_avoided_losses')
    mean_possible_avoided_losses = (upper_bound_possible_avoided_losses + lower_bound_possible_avoided_losses) / 2
    diff_possible_avoided_losses = (upper_bound_possible_avoided_losses - lower_bound_possible_avoided_losses) / 2

    global_avoided_losses = pd.concat([upper_bound_avoided_losses, mean_avoided_losses, lower_bound_avoided_losses, diff_avoided_losses],
                                      keys=['upper_bound', 'mean', 'lower_bound', 'yerr'])
    global_possible_avoided_losses = pd.concat([upper_bound_possible_avoided_losses, mean_possible_avoided_losses, lower_bound_possible_avoided_losses, diff_possible_avoided_losses],
                                        keys=['upper_bound', 'mean', 'lower_bound', 'yerr'])

    plot_data = global_avoided_losses.copy()
    plot_data[None] = 0
    plot_data = pd.concat([plot_data, global_possible_avoided_losses.rename('perfect')], axis=1)
    plot_data.columns.name = 'year'
    plot_data = plot_data.groupby(level=[0, 2]).sum() / 1e9
    plot_data = plot_data.T

    fig, ax = plt.subplots(figsize=(6, 4))
    plot_data.plot.bar(y='mean', width=.75, ax=ax, color=[plt.get_cmap('Blues')(.5), plt.get_cmap('Greens')(.5)],
                       alpha=.75, yerr='yerr', error_kw=dict(ecolor='gray', alpha=.5))
    ax.set_xticks(list(np.arange(len(plot_data) - 2)) + [len(plot_data) - 1])
    ax.set_xticklabels([int(y) if not np.isnan(y) else '' for y in plot_data.index[:-2]] + ['perfect'])
    ax.set_xlabel('year')
    ax.set_ylabel('Avoided losses (bn $PPP)')
    plt.tight_layout()

    if outpath_:
        plt.savefig(f"{outpath_}/avoided_losses_over_time.pdf", dpi=300, bbox_inches='tight')

    # print avoided losses over the entire time
    upper_bound_loss_gap = pd.Series(
        data=upper_bound_results[np.max(upper_bound_results.columns)].values - full_ew_results.values,
        index=upper_bound_results.index,
        name='upper_bound_loss_gap')
    lower_bound_loss_gap = pd.Series(
        data=lower_bound_results[np.max(lower_bound_results.columns)].values - full_ew_results.values,
        index=lower_bound_results.index,
        name='lower_bound_loss_gap')
    mean_loss_gap = (upper_bound_loss_gap + lower_bound_loss_gap) / 2
    print("The loss gap to perfect early warning is (mean, upper, lower):", mean_loss_gap.groupby(level=1).sum() / 1e9,
          upper_bound_loss_gap.groupby(level=1).sum() / 1e9,
          lower_bound_loss_gap.groupby(level=1).sum() / 1e9)

    print("Global avoided losses over the entire time (min, max): \n", global_avoided_losses.loc['lower_bound'].groupby(level=1).sum().sum(axis=1) / 1e9, global_avoided_losses.loc['upper_bound'].groupby(level=1).sum().sum(axis=1) / 1e9)


def plot_drivers_gdp_and_gini_index(dataset, outpath=None, annotate=None):
    income_groups_ = load_income_groups()
    merged = pd.merge(dataset, income_groups_, left_on='iso3', right_index=True, how='left')
    merged.replace({'High income': 'High', 'Upper middle income': 'Upper middle',
                    'Lower middle income': 'Lower middle', 'Low income': 'Low'}, inplace=True)
    merged['income_group_order'] = merged['Country income group'].replace({'Low': 0, 'Lower middle': 1,
                                                                     'Upper middle': 2, 'High': 3})
    merged = merged.sort_values('income_group_order').reset_index(drop=True)
    fig1, ax1 = plt.subplots(figsize=(4.5, 4.3))
    sns.boxplot(merged, x='Country income group', y='resilience', palette='tab10')
    ax1.set_ylabel('resilience (%)')
    plt.tight_layout()

    # plot resilience vs GDP per capita, coloring by Country income group
    fig2, ax2 = plt.subplots(figsize=(4.5, 4.3))
    sns.scatterplot(data=merged, x='gdp_pc_pp', y='resilience', hue='Country income group', palette='tab10', alpha=.5)
    ax2.set_xlabel('GDP per capita ($PPP)')
    ax2.set_ylabel('resilience (%)')
    plt.tight_layout()
    fig3, ax3 = plt.subplots(figsize=(4.5, 4.3))
    sns.scatterplot(data=merged, x='gdp_pc_pp', y='resilience', alpha=.5)
    ax3.set_xlabel('GDP per capita ($PPP)')
    ax3.set_ylabel('resilience (%)')
    plt.tight_layout()

    # plot resilience vs gini index
    fig4, ax4 = plt.subplots(figsize=(4.5, 4.3))
    sns.scatterplot(data=merged, x='gini_index', y='resilience', alpha=.5)#, hue='Country income group', palette='tab10')
    ax4.set_xlabel('Gini index (%)')
    ax4.set_ylabel('resilience (%)')
    plt.tight_layout()
    if outpath:
        fig4.savefig(os.path.join(outpath, "resilience_vs_gini_index_scatter_not_annotated.pdf"), dpi=300)

    # plot Risk to assets vs risk to assets
    fig5, ax5 = plt.subplots(figsize=(4.5, 4.3))
    sns.scatterplot(data=merged, x='gdp_pc_pp', y='risk_to_assets', alpha=.5)  # , hue='Country income group', palette='tab10')
    ax5.set_xlabel('GDP per capita ($PPP)')
    ax5.set_ylabel('risk to assets (% GDP)')
    plt.tight_layout()

    # plot Risk vs risk to assets
    fig6, ax6 = plt.subplots(figsize=(4.5, 4.3))
    sns.scatterplot(data=merged, x='gdp_pc_pp', y='risk', alpha=.5)  # , hue='Country income group', palette='tab10')
    ax6.set_xlabel('GDP per capita ($PPP)')
    ax6.set_ylabel('risk to wellbeing (% GDP)')
    plt.tight_layout()

    if annotate:
        for i, row in merged.iterrows():
            if row.iso3 in annotate:
                ax2.annotate(row.iso3, (row.gdp_pc_pp, row.resilience))
                ax3.annotate(row.iso3, (row.gdp_pc_pp, row.resilience))
                ax4.annotate(row.iso3, (row.gini_index, row.resilience))
                ax5.annotate(row.iso3, (row.gdp_pc_pp, row.risk_to_assets))
                ax6.annotate(row.iso3, (row.gdp_pc_pp, row.risk))
    if outpath:
        fig1.savefig(os.path.join(outpath, "resilience_income_groups_boxplot.pdf"), dpi=300)
        fig2.savefig(os.path.join(outpath, "resilience_vs_gdp_pc_scatter_color.pdf"), dpi=300)
        fig3.savefig(os.path.join(outpath, "resilience_vs_gdp_pc_scatter_no_color.pdf"), dpi=300)
        fig5.savefig(os.path.join(outpath, "risk_to_assets_vs_gdp_pc_pp_scatter.pdf"), dpi=300)
        fig4.savefig(os.path.join(outpath, "resilience_vs_gini_index_scatter_annotated.pdf"), dpi=300)
        fig6.savefig(os.path.join(outpath, "risk_to_wellbeing_vs_gdp_pc_pp_scatter.pdf"), dpi=300)


def make_liquidity_comparison_plot(path_with, path_without, is_pds=False, outpath_=None):
    cat_info_with_liquidity = pd.read_csv(path_with + ("iah_tax_unif_poor.csv" if is_pds else "/iah_tax_no.csv"), index_col=[0, 1, 2, 3, 4, 5])
    cat_info_with_liquidity['t_reco_95'] = cat_info_with_liquidity.lambda_h.apply(lambda x: np.log(1 / .05) / x)
    cat_info_without_liquidity = pd.read_csv(path_without + "/iah_tax_no.csv", index_col=[0, 1, 2, 3, 4, 5])
    cat_info_without_liquidity['t_reco_95'] = cat_info_without_liquidity.lambda_h.apply(lambda x: np.log(1 / .05) / x)
    cat_info_merged = pd.merge(cat_info_with_liquidity, cat_info_without_liquidity, left_index=True, right_index=True, suffixes=('_with', '_without'))
    cat_info_merged = pd.merge(cat_info_merged, income_groups.replace({'Low income': 'Low', 'Lower middle income': 'Lower middle  ', 'Upper middle income': '  Upper middle', 'High income': 'High'}), left_on='iso3', right_index=True, how='left')
    results_with_liquidity = pd.read_csv(path_with + ("results_tax_unif_poor.csv" if is_pds else "/results_tax_no.csv"), index_col=0)
    results_with_liquidity[['resilience', 'risk', 'risk_to_assets']] *= 100
    results_without_liquidity = pd.read_csv(path_without + "/results_tax_no.csv", index_col=0)
    results_without_liquidity[['resilience', 'risk', 'risk_to_assets']] *= 100
    results_merged = pd.merge(results_with_liquidity, results_without_liquidity, left_index=True, right_index=True, suffixes=('_with', '_without'))
    results_merged = pd.merge(results_merged, income_groups.replace({'Low income': 'Low', 'Lower middle income': 'Lower middle  ', 'Upper middle income': '  Upper middle', 'High income': 'High'}), left_on='iso3', right_index=True, how='left')
    results_merged['Avoided wellbeing losses (%)'] = (results_merged.dWtot_currency_without - results_merged.dWtot_currency_with) / results_merged.dWtot_currency_without * 100

    if not is_pds:
        rename = {
            'gdp_pc_pp_with': 'GDP per capita',
            'resilience_without': 'without liquid savings',
            'resilience_with': 'with liquid savings',
        }
    else:
        rename = {
            'gdp_pc_pp_with': 'GDP per capita',
            'resilience_without': 'without PDS',
            'resilience_with': 'with PDS',
        }
    fig, ax = plt.subplots(figsize=(4.5, 4.3))
    sns.scatterplot(data=results_merged.rename(columns=rename),
                    x=rename['resilience_without'], y=rename['resilience_with'], alpha=.5,
                    size='GDP per capita', size_norm=(20000, 100000), legend='auto')
    ax.axline([35, 35], slope=1, ls='--', c='k', label='y=x', alpha=0.5)
    ax.set_title('Socio-economic resilience (%)')
    plt.tight_layout()
    if outpath_:
        plt.savefig(os.path.join(outpath_, f"{'pds' if is_pds else 'liquidity'}_comparison_scatter.pdf"), dpi=300, bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(4.5, 4.3))
    sns.boxplot(data=results_merged.rename(columns=rename), x='Country income group', y='Avoided wellbeing losses (%)', palette='tab10',
                order=['Low', 'Lower middle  ', '  Upper middle', 'High'])
    ax.set_title('Avoided wellbeing losses (%)')
    plt.tight_layout()
    if outpath_:
        plt.savefig(os.path.join(outpath_, f"{'pds' if is_pds else 'liquidity'}_avoided_wellbeing_losses_boxplot.pdf"), dpi=300, bbox_inches='tight')

    if not is_pds:
        rename = {
            'gdp_pc_pp_with': 'GDP per capita',
            't_reco_95_with': 'with liquid savings',
            't_reco_95_without': 'without liquid savings',
        }
    else:
        rename = {
            'gdp_pc_pp_with': 'GDP per capita',
            't_reco_95_with': 'with PDS',
            't_reco_95_without': 'without PDS',
        }
    cat_info_merged = cat_info_merged[cat_info_merged.t_reco_95_with <= cat_info_merged.t_reco_95_without]
    if is_pds:
        cat_info_merged = cat_info_merged[cat_info_merged.help_received_with > 0]
    fig, ax = plt.subplots(figsize=(4.5, 4.3))
    sns.scatterplot(data=cat_info_merged.rename(columns=rename),
                    x=rename['t_reco_95_without'], y=rename['t_reco_95_with'], alpha=.5)
    ax.axline([0, 0], slope=1, ls='--', c='k', label='y=x', alpha=0.5)
    ax.set_title('Time to recover 95% of initial asset losses (y)')
    plt.tight_layout()
    plt.legend()
    if outpath_:
        plt.savefig(os.path.join(outpath_, f"{'pds' if is_pds else 'liquidity'}_recovery_time_comparison_scatter.pdf"), dpi=300, bbox_inches='tight')


def make_liquid_savings_distribution_boxplot():
    liquidity = pd.read_csv("./inputs/FINDEX/findex_liquidity.csv", index_col=[0, 1, 2])
    liquidity = liquidity.iloc[liquidity.reset_index().groupby(['iso3', 'income_cat']).year.idxmax()][['liquidity_share', 'liquidity']].prod(axis=1)
    liquidity = liquidity.droplevel('year')

    asset_losses = pd.read_csv("./output/scenarios/Existing_climate/2024-05-03_14-09_no_liquidity_EW-2018_noPDS/iah_tax_no.csv", index_col=[0, 1, 2, 3, 4, 5])[['dk', 'n']]
    asset_losses = average_over_rp(agg_to_event_level(asset_losses, 'dk', ['iso3', 'hazard', 'rp', 'income_cat']), None).groupby(['iso3', 'income_cat']).sum()

    merged = pd.merge(asset_losses.rename('Asset loss'), liquidity.rename('liquid savings'), left_index=True, right_index=True, how='inner')
    merged = pd.merge(merged, income_groups, left_on='iso3', right_index=True, how='left')
    merged['Asset losses as share of liquid savings (%)'] = merged['Asset loss'] / merged['liquid savings'] * 100
    merged.index.names = ['iso3', 'Household income quintile']
    merged.replace({'High income': 'High', 'Upper middle income': 'Upper middle', 'Lower middle income': 'Lower middle', 'Low income': 'Low'}, inplace=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=merged, x='Country income group', y='Asset losses as share of liquid savings (%)', palette='tab10', hue='Household income quintile',
                fliersize=0, order=['Low', 'Lower middle', 'Upper middle', 'High'])
    ax.legend(loc='upper left', title='Household income quintile', bbox_to_anchor=(1, 1))


def plot_recovery(t_max, productivity_pi_, delta_tax_sp_, k_h_eff_, delta_k_h_eff_, lambda_h_, sigma_h_, savings_s_h_,
                  delta_i_h_pds_, delta_c_h_max_, recovery_params_, social_protection_share_gamma_h_, diversified_share_,
                  show_sp_losses=False, consumption_floor_xi_=None, t_hat_=None, t_tilde_=None, delta_tilde_k_h_eff_=None,
                  consumption_offset_=None, title=None, ylims=None, plot_legend=True, show_ylabel=True, fig=None, axs=None,
                  plot_capital=True, linecolor='black', shading_color='red'):
    """
    Make a plot of the consumption and capital losses over time
    """
    if fig is None and axs is None:
        if plot_capital:
            fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(7, 5))
        else:
            fig, axs = plt.subplots(nrows=1, sharex=True, figsize=(7, 2.5))
            axs = [axs]

    t_ = np.linspace(0, t_max, 1000)
    if t_tilde_ is not None and t_tilde_ not in t_:
        t_ = np.array(sorted(list(t_) + [t_tilde_]))
    if t_hat_ is not None and t_hat_ not in t_:
        t_ = np.array(sorted(list(t_) + [t_hat_]))
    if t_hat_ is not None and t_tilde_ is not None and t_hat_ + t_tilde_ not in t_:
        t_ = np.array(sorted(list(t_) + [t_hat_ + t_tilde_]))
    c_baseline = baseline_consumption_c_h(productivity_pi_, k_h_eff_, delta_tax_sp_, diversified_share_)
    di_h_lab, di_h_sp, dc_reco, dc_savings_pds = delta_c_h_of_t(t_, productivity_pi_, delta_tax_sp_, delta_k_h_eff_, lambda_h_,
                                                                sigma_h_, savings_s_h_, delta_i_h_pds_, delta_c_h_max_,
                                                                recovery_params_, social_protection_share_gamma_h_,
                                                                consumption_floor_xi_, t_hat_, t_tilde_,
                                                                delta_tilde_k_h_eff_, consumption_offset_,
                                                                True)
    di_h = di_h_lab + di_h_sp
    if show_sp_losses:
        axs[0].fill_between(t_, c_baseline, c_baseline - di_h_sp, color=shading_color, alpha=0.75, label='Transfers loss', lw=0)
    axs[0].fill_between(t_, c_baseline - di_h_sp, c_baseline - di_h, color=shading_color, alpha=0.5,
                       label='Income loss', lw=0)
    axs[0].fill_between(t_, c_baseline - di_h, c_baseline - (di_h + dc_reco), color=shading_color, alpha=0.25,
                        label='Reconstruction loss', lw=0)
    axs[0].fill_between(t_[dc_savings_pds != 0], (c_baseline - (di_h + dc_reco) + dc_savings_pds)[dc_savings_pds != 0],
                       (c_baseline - (di_h + dc_reco))[dc_savings_pds != 0], facecolor='none', lw=0, hatch='XXX',
                        edgecolor='grey', label='Liquidity')
    axs[0].plot([-0.03 * (max(t_) - min(t_)), 0], [c_baseline, c_baseline], color=linecolor, label='__none__')
    axs[0].plot([0, 0], [c_baseline, (c_baseline - di_h - dc_reco + dc_savings_pds)[0]], color=linecolor, label='__none__')
    axs[0].plot(t_, c_baseline - di_h - dc_reco + dc_savings_pds, color=linecolor, label='Consumption')

    dk_eff = delta_k_h_eff_of_t(t_, 0, delta_k_h_eff_, lambda_h_, sigma_h_, delta_c_h_max_, productivity_pi_)

    if plot_capital:
        axs[1].plot([-0.03 * (max(t_) - min(t_)), 0], [0, 0], color=linecolor, label='__none__')
        axs[1].plot([0, 0], [0, dk_eff[0]], color=linecolor, label='__none__')
        axs[1].plot(t_, dk_eff, color=linecolor, label='Effective capital loss')

    axs[-1].set_xlabel('Time [yr]')

    if show_ylabel:
        axs[0].set_ylabel('Consumption [$PPP]')
        if plot_capital:
            axs[1].set_ylabel('Capital loss [$PPP]')
    else:
        axs[0].set_ylabel(None)
        axs[0].set_yticklabels([])
        if plot_capital:
            axs[1].set_ylabel(None)
            axs[1].set_yticklabels([])

    if ylims is not None:
        axs[0].set_ylim(ylims[0])
        if plot_capital:
            axs[1].set_ylim(ylims[1])

    if title is not None:
        axs[0].set_title(title)
    if plot_legend:
        for ax in axs:
            ax.legend(frameon=False, bbox_to_anchor=(1, 1), loc='upper left')
    plt.tight_layout()


def plot_supfigs_3_4(results_data_, outpath_=None):
    capital_shares = results_data_.copy()
    fig, axs = plt.subplots(ncols=3, nrows=3, figsize=(double_col_width * centimeter, 16 * centimeter), sharex=False, sharey='row')
    capital_shares['owner_occ_share_of_value_added'] = capital_shares['real_estate_share_of_value_added'] * capital_shares['home_ownership_rate']
    capital_shares[['k_pub_share', 'k_priv_share', 'k_household_share', 'real_estate_share_of_value_added', 'home_ownership_rate', 'self_employment', 'owner_occ_share_of_value_added']] *= 100
    capital_shares['gdp_pc_pp'] /= 1e3

    for ax, (x, y), name in zip(axs[0, :], [('gdp_pc_pp', 'k_pub_share'), ('gdp_pc_pp', 'k_priv_share'), ('gdp_pc_pp', 'k_household_share')], [r'$\kappa^p$', r'$\kappa^f$', r'$\kappa^h$']):
        legend = False
        if ax == axs[0, -1]:
            legend = True
        sns.scatterplot(capital_shares, x=x, y=y, ax=ax, alpha=.5, s=10, hue='Country income group', hue_order=['LICs', 'LMICs', 'UMICs', 'HICs'], legend=legend, palette=INCOME_GROUP_COLORS,
                                   style='Country income group', markers=INCOME_GROUP_MARKERS)
        for label in capital_shares.index:
            ax.text(capital_shares[x].loc[label], capital_shares[y].loc[label], label, fontsize=6)
        ax.set_xlabel('GDP per capita [$1,000 PPP]')
        ax.set_title(name)
    axs[0, -1].legend(frameon=False, bbox_to_anchor=(1, 1), loc='upper left')
    axs[0, 0].set_ylabel('share [%]')

    for ax, (x, y), name in zip(axs[1, :], [('self_employment', 'k_pub_share'), ('self_employment', 'k_priv_share'), ('self_employment', 'k_household_share')], [r'$\kappa^p$', r'$\kappa^f$', r'$\kappa^h$']):
        x_ = capital_shares[x]
        sns.scatterplot(capital_shares, x=x, y=y, ax=ax, alpha=.5, s=10, hue='Country income group', hue_order=['LICs', 'LMICs', 'UMICs', 'HICs'], legend=False, palette=INCOME_GROUP_COLORS,
                                   style='Country income group', markers=INCOME_GROUP_MARKERS)
        for label in capital_shares.index:
            ax.text(x_.loc[label], capital_shares[y].loc[label], label, fontsize=6)
            ax.set_xlabel('self employment rate [%]')
        axs[1, 0].set_ylabel('share [%]')

    # for ax, (x, y), name in zip(axs[2, :], [('home_ownership_rate', 'k_pub_share'), ('home_ownership_rate', 'k_priv_share'), ('home_ownership_rate', 'k_household_share')], [r'$\kappa^p$', r'$\kappa^f$', r'$\kappa^h$']):
    for ax, (x, y), name in zip(axs[2, :], [('owner_occ_share_of_value_added', 'k_pub_share'), ('owner_occ_share_of_value_added', 'k_priv_share'), ('owner_occ_share_of_value_added', 'k_household_share')], [r'$\kappa^p$', r'$\kappa^f$', r'$\kappa^h$']):
    # for ax, (x, y), name in zip(axs[2, :], [('real_estate_share_of_value_added', 'k_pub_share'), ('real_estate_share_of_value_added', 'k_priv_share'), ('real_estate_share_of_value_added', 'k_household_share')], [r'$\kappa^p$', r'$\kappa^f$', r'$\kappa^h$']):
        x_ = capital_shares[x]
        sns.scatterplot(capital_shares, x=x, y=y, ax=ax, alpha=.5, s=10, hue='Country income group', hue_order=['LICs', 'LMICs', 'UMICs', 'HICs'], legend=False, palette=INCOME_GROUP_COLORS,
                                   style='Country income group', markers=INCOME_GROUP_MARKERS)
        for label in capital_shares.index:
            ax.text(x_.loc[label], capital_shares[y].loc[label], label, fontsize=6)
            # ax.set_xlabel('Home ownership rate [%]')
            ax.set_xlabel('Owner-occupied share\nof value added [%]')
            # ax.set_xlabel('Real-estate share of GDP [%]')
        axs[2, 0].set_ylabel('share [%]')

    plt.tight_layout()
    if outpath_ is not None:
        fig.savefig(os.path.join(outpath_, f"supfig_3.pdf"), dpi=300, bbox_inches='tight')
    plt.show(block=False)

    fig, axs = plt.subplots(figsize=(2 * single_col_width * centimeter, single_col_width * centimeter), ncols=2)
    sns.scatterplot(data=capital_shares, x='gdp_pc_pp', y='self_employment', alpha=.5, hue='Country income group', hue_order=['LICs', 'LMICs', 'UMICs', 'HICs'], palette=INCOME_GROUP_COLORS,
                    style='Country income group', markers=INCOME_GROUP_MARKERS, s=10, ax=axs[0], legend=False)
    axs[0].legend(frameon=False, bbox_to_anchor=(1, 1), loc='upper left')
    axs[0].set_xlabel('GDP per capita [$1,000 PPP]')
    axs[0].set_ylabel('self employment rate [%]')

    # sns.scatterplot(data=capital_shares, x='gdp_pc_pp', y='home_ownership_rate', alpha=.5, hue='Country income group',
    sns.scatterplot(data=capital_shares, x='gdp_pc_pp', y='owner_occ_share_of_value_added', alpha=.5, hue='Country income group',
                    hue_order=['LICs', 'LMICs', 'UMICs', 'HICs'], palette=INCOME_GROUP_COLORS,
                    style='Country income group', markers=INCOME_GROUP_MARKERS, s=10, ax=axs[1])
    axs[1].legend(frameon=False, bbox_to_anchor=(1, 1), loc='upper left')
    axs[1].set_xlabel('GDP per capita [$1,000 PPP]')
    # axs[1].set_ylabel('home ownership rate [%]')
    axs[1].set_ylabel('Owner-occupied share of value added [%]')
    plt.tight_layout()
    if outpath_ is not None:
        fig.savefig(os.path.join(outpath_, f"supfig_4.pdf"), dpi=300, bbox_inches='tight')
    plt.show(block=False)


def plot_fig_1(cat_info_data_, macro_data_, countries, hazard='Flood', plot_rp=100, outpath=None):
    if type(countries) == str:
        countries = [countries]
    plot_data = pd.merge(
        cat_info_data_.loc[pd.IndexSlice[countries, hazard, plot_rp, :, :, :], :],
        macro_data_.drop(['fa'], axis=1),
        left_index=True,
        right_index=True,
    )

    plot_data['dw_currency'] = plot_data.dw / (plot_data.gdp_pc_pp**(-plot_data.income_elasticity_eta))
    plot_data[['c', 'k', 'gdp_pc_pp', 'dk', 'dw_currency']] /= 1e3
    plot_data['k_non_hh'] = plot_data.k * (1 - plot_data.k_household_share)
    plot_data['dk_non_hh'] = plot_data.dk * (1 - plot_data.k_household_share)
    plot_data['c_labor'] = plot_data.k * plot_data.avg_prod_k * (1 - plot_data.tau_tax)
    plot_data[['v_ew', 'fa']] *= 100

    print("Characteristics:\n", plot_data.loc[pd.IndexSlice[:, :, :, :, 'a', 'not_helped'], ['fa', 't_reco_95']])
    print("Asset losses [bn USD]:\n", plot_data[['dk', 'n', 'pop']].prod(axis=1).groupby('iso3').sum() * 1e3 / 1e9)
    print("Wellbeing losses [bn USD]:\n", plot_data[['dw_currency', 'n', 'pop']].prod(axis=1).groupby('iso3').sum() * 1e3 / 1e9)

    fig1_1 = plot_fig_1_1(plot_data)
    fig1_2 = plot_fig_1_2(plot_data)
    fig1_2.subplots_adjust(top=1 - .8 * centimeter / fig1_2.get_figheight())
    fig1_2.text(0, 1, ' ', ha='left', va='top')


    if outpath:
        fig1_1.savefig(outpath + '/fig_1_1.pdf', dpi=300, bbox_inches='tight', transparent=True, pad_inches=0)
        fig1_2.savefig(outpath + '/fig_1_2.pdf', dpi=300, bbox_inches='tight', transparent=True, pad_inches=0)
    plt.show(block=False)


def plot_fig_1_2(plot_data):
    fig, axs = plt.subplots(ncols=2, figsize=(double_col_width * centimeter, 2), sharey=False, sharex=True)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:2]
    for country, ax, legend, c in zip(plot_data.index.get_level_values('iso3').unique(), axs, [False, True], colors):
        country_data = plot_data.loc[pd.IndexSlice[country, :, :, 'q5', 'a', 'not_helped']].iloc[0]
        title = f"{country}-q5"
        recovery_params = [(k / 1e3, l) for (k, l) in country_data.recovery_params]
        plot_recovery(3, country_data.avg_prod_k, country_data.tau_tax, country_data.k, country_data.dk,
                      country_data.lambda_h, country_data.k_household_share, country_data.liquidity / 1e3,
                      0, np.nan, recovery_params, country_data.gamma_SP * country_data.n,
                      country_data.diversified_share, axs=[ax], show_ylabel=not legend, plot_capital=False,
                      plot_legend=False, linecolor=c, shading_color='dimgrey',
                      ylims=[(0, 10.000), None], title=title)
    axs[0].set_ylabel('Consumption [$PPP 1,000]')
    legend = axs[-1].legend(frameon=False, bbox_to_anchor=(1, 1), loc='upper left')
    handles = legend.legend_handles
    handles[-1].set_color('k')
    for i, ax in enumerate(axs):
        ax.text(0, 1, string.ascii_lowercase[i+8], transform=ax.transAxes, fontsize=8, fontweight='bold', ha='right',
                va='bottom')
    plt.tight_layout()
    return fig


def plot_fig_1_1(plot_data):
    fig, axs = plt.subplots(ncols=4, nrows=2, figsize=(double_col_width * centimeter, 4), sharex=False)

    # plot pre-disaster effective capital
    ax = axs[0, 0]
    sns.barplot(
        data=plot_data.loc[pd.IndexSlice[:, :, :, :, 'a', 'not_helped']],
        x='income_cat', y='k', alpha=.5, hue='iso3', ax=ax, legend=False
    )
    # plot pre-disaster non-hh-owned capital
    sns.barplot(
        data=plot_data.loc[pd.IndexSlice[:, :, :, :, 'a', 'not_helped']],
        x='income_cat', y='k_non_hh', alpha=1, hue='iso3', ax=ax, legend=False
    )
    sns.barplot(
        data=plot_data.loc[pd.IndexSlice[:, :, :, :, 'a', 'not_helped']].assign(k=0),
        x='income_cat', y='k', alpha=.5, ax=ax, color='darkgrey', label='hh-owned', linewidth=0
    )
    sns.barplot(
        data=plot_data.loc[pd.IndexSlice[:, :, :, :, 'a', 'not_helped']].assign(k=0),
        x='income_cat', y='k', alpha=1, ax=ax, color='darkgrey', label='other', linewidth=0
    )
    ax.legend(frameon=False)
    ax.set_ylabel(None)
    ax.set_title('Assets\n[$PPP 1,000]')

    # plot pre-disaster total income
    ax = axs[0, 1]
    sns.barplot(
        data=plot_data.loc[pd.IndexSlice[:, :, :, :, 'a', 'not_helped']],
        x='income_cat', y='c', ax=ax, hue='iso3', alpha=.5, legend=False
    )
    # plot pre-disaster labor income
    sns.barplot(
        data=plot_data.loc[pd.IndexSlice[:, :, :, :, 'a', 'not_helped']],
        x='income_cat', y='c_labor', ax=ax, hue='iso3', legend=False
    )
    sns.barplot(
        data=plot_data.loc[pd.IndexSlice[:, :, :, :, 'a', 'not_helped']].assign(c_labor=0),
        x='income_cat', y='c_labor', alpha=.5, ax=ax, color='darkgrey', label='diversified', linewidth=0
    )
    sns.barplot(
        data=plot_data.loc[pd.IndexSlice[:, :, :, :, 'a', 'not_helped']].assign(c_labor=0),
        x='income_cat', y='c_labor', alpha=1, ax=ax, color='darkgrey', label='labor', linewidth=0
    )
    ax.legend(frameon=False)
    ax.set_ylabel(None)
    ax.set_title('Income\n[$PPP 1,000]')

    # plot liquidity data
    ax = axs[0, 2]
    sns.barplot(
        data=plot_data.loc[pd.IndexSlice[:, :, :, :, 'a', 'not_helped']],
        x='income_cat', y='liquidity', ax=ax, hue='iso3', legend=False
    )
    ax.set_ylabel(None)
    ax.set_title('Liquidity\n[$PPP]')

    # plot vulnerability
    ax = axs[0, 3]
    sns.barplot(
        data=plot_data.loc[pd.IndexSlice[:, :, :, :, 'a', 'not_helped']],
        x='income_cat', y='v_ew', ax=ax, hue='iso3', legend=True
    )
    ax.set_ylabel(None)
    ax.set_title('Vulnerability\n[%]')
    ax.legend(frameon=False, title=False, bbox_to_anchor=(1, 1), loc='upper left')

    # plot total asset losses
    ax = axs[1, 0]
    sns.barplot(
        data=plot_data.loc[pd.IndexSlice[:, :, :, :, 'a', 'not_helped']],
        x='income_cat', y='dk', ax=ax, hue='iso3', legend=False, alpha=.5
    )
    # plot not hh-owned asset losses
    sns.barplot(
        data=plot_data.loc[pd.IndexSlice[:, :, :, :, 'a', 'not_helped']],
        x='income_cat', y='dk_non_hh', ax=ax, hue='iso3', legend=False
    )
    sns.barplot(
        data=plot_data.loc[pd.IndexSlice[:, :, :, :, 'a', 'not_helped']].assign(dk_non_hh=0),
        x='income_cat', y='dk_non_hh', alpha=.5, ax=ax, color='darkgrey', label='hh-owned', linewidth=0
    )
    sns.barplot(
        data=plot_data.loc[pd.IndexSlice[:, :, :, :, 'a', 'not_helped']].assign(dk_non_hh=0),
        x='income_cat', y='dk_non_hh', alpha=1, ax=ax, color='darkgrey', label='other', linewidth=0
    )
    ax.legend(frameon=False)
    ax.set_ylabel(None)
    ax.set_title('Asset losses\n[$PPP 1,000]')

    # plot recovery time
    ax = axs[1, 1]
    sns.barplot(
        data=plot_data.loc[pd.IndexSlice[:, :, :, :, 'a', 'not_helped']].t_reco_95.unstack('iso3').assign(
            TJK=0).stack().rename('t_reco_95').to_frame(),
        x='income_cat', y='t_reco_95', ax=ax, hue='iso3', legend=False
    )
    # plot recovery time
    reco_twnx = ax.twinx()
    sns.barplot(
        data=plot_data.loc[pd.IndexSlice[:, :, :, :, 'a', 'not_helped']].t_reco_95.unstack('iso3').assign(
            HTI=0).stack().rename('t_reco_95').to_frame(),
        x='income_cat', y='t_reco_95', ax=reco_twnx, hue='iso3', legend=False
    )
    reco_twnx.set_ylabel(None)
    ax.set_ylabel(None)
    ax.set_title('Recovery time\n[yr]')

    # plot wellbeing losses of exposed households
    ax = axs[1, 2]
    sns.barplot(
        data=plot_data.loc[pd.IndexSlice[:, :, :, :, 'a', 'not_helped']],
        x='income_cat', y='dw_currency', ax=ax, hue='iso3', legend=False
    )
    ax.set_ylabel(None)
    ax.set_title('Wellbeing losses, affected\n[$PPP 1,000]')

    # plot wellbeing losses of exposed households
    ax = axs[1, 3]
    sns.barplot(
        data=plot_data.loc[pd.IndexSlice[:, :, :, :, 'na', 'not_helped']],
        x='income_cat', y='dw_currency', ax=ax, hue='iso3', legend=False
    )
    ax.set_ylabel(None)
    ax.set_title('Wellbeing losses, non-affected\n[$PPP 1,000]')

    plt.tight_layout()

    for i, ax in enumerate(axs.flatten()):
        ax.text(-.05, 1.0, string.ascii_lowercase[i], transform=ax.transAxes, fontsize=8, fontweight='bold', ha='right',
                va='bottom')

    for ax in axs[0, :]:
        ax.set_xlabel(None)

    for ax in axs[1, :]:
        ax.set_xlabel(None)
    fig.text(0.5, 0.03, 'Household income quintile', ha='center', va='bottom')

    return fig


def compute_national_recovery_duration(cat_info_data_, outpath=None):
    """
    Compute the national recovery duration based on the recovery parameters
    """
    result = cat_info_data_.xs('a', level='affected_cat')[['k', 'dk', 'n']]
    result['k_tot'] = result.k * result.n
    result['dk_tot'] = result.dk * result.n
    result = result.groupby(['iso3', 'hazard', 'rp']).dk_tot.sum() / result.groupby(['iso3', 'hazard', 'rp']).k_tot.sum() * 100
    result = result.rename('capital_loss_rel').to_frame()

    for idx in tqdm.tqdm(result.index, desc='Computing national recovery duration'):
        idx_data = cat_info_data_.loc[idx].xs('a', level='affected_cat')
        dk_tot = idx_data[['dk', 'n']].prod(axis=1)

        if dk_tot.sum() == 0:
            result.loc[idx, 't_reco_95'] = np.nan
            continue

        def fun(t, full_reco=.95):
            # the function value
            dk_of_t = dk_tot * np.exp(-idx_data.lambda_h * t) / dk_tot.sum()

            # the derivative
            ddk_of_t = -idx_data.lambda_h * dk_of_t

            # the second derivative
            d2dk_of_t = idx_data.lambda_h**2 * dk_of_t

            return dk_of_t.sum() - (1 - full_reco), ddk_of_t.sum(), d2dk_of_t.sum()

        # find the time to recover 95% of the asset losses
        t_reco_95 = root_scalar(fun, fprime=True, fprime2=True, x0=0, bracket=(0, 1e5 )).root
        result.loc[idx, 't_reco_95'] = t_reco_95
        # result.loc[idx, 'probability'] = 1 / idx[2]

    if outpath:
        result.to_csv(outpath)
    return result#, durations


def load_data(simulation_paths_, model_root_dir_):
    gini_index_ = pd.read_csv(os.path.join(model_root_dir_, "inputs/raw/WB_socio_economic_data/gini_index.csv"), index_col=0)

    income_groups_ = load_income_groups(model_root_dir_)

    cat_info_data_ = {
        k: pd.read_csv(os.path.join(v, "iah.csv"), index_col=[0, 1, 2, 3, 4, 5]) for k, v in
        simulation_paths_.items()
    }

    for k in cat_info_data_.keys():
        cat_info_data_[k]['t_reco_95'] = np.log(1 / .05) / cat_info_data_[k].lambda_h
        cat_info_data_[k].recovery_params = cat_info_data_[k].recovery_params.apply(lambda x: [(float(d.split(', ')[0]), float(d.split(', ')[1])) for d in x[2:-2].split('), (')])

    macro_data_ = {
        k: pd.read_csv(os.path.join(v, "macro.csv"), index_col=[0, 1, 2]) for k, v in
        simulation_paths_.items()
    }

    results_data_ = {
        k: pd.read_csv(os.path.join(v, "results.csv"), index_col=0) for k, v in
        simulation_paths_.items()
    }

    for k in results_data_.keys():
        results_data_[k] = results_data_[k].drop('THA')
        results_data_[k][['resilience', 'risk', 'risk_to_assets']] *= 100
        results_data_[k] = results_data_[k].join(gini_index_, on='iso3')
        results_data_[k]['log_gdp_pc_pp'] = np.log(results_data_[k]['gdp_pc_pp'])
        results_data_[k] = pd.merge(results_data_[k], income_groups_, left_on='iso3', right_index=True, how='left')

    name_dict_ = {
        'resilience': 'socio-economic resilience [%]',
        'risk': 'risk to wellbeing [% of GDP]',
        'risk_to_assets': 'risk to assets [% of GDP]',
        'gdp_pc_pp': 'GDP per capita [$PPP]',
        'dk_tot': 'Asset losses [$PPP]',
        'dWtot_currency': 'Welfare losses [$PPP]',
        'gini_index': 'Gini index [%]',
    }
    any_to_wb_, iso3_to_wb_, iso2_iso3_ = get_country_name_dicts("./")

    return income_groups_, gini_index_, cat_info_data_, macro_data_, results_data_, name_dict_, any_to_wb_, iso3_to_wb_, iso2_iso3_


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script parameters')
    parser.add_argument('simulation_outputs_dir', type=str)
    parser.add_argument('outpath', type=str)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()

    outpath = args.outpath
    os.makedirs(outpath, exist_ok=True)
    model_root_dir = os.path.dirname(os.path.abspath(__file__))

    simulation_paths = {
        'baseline': '0_baseline',
        'reduce_total_exposure_0.05': '1_reduce_total_exposure/q1+q2+q3+q4+q5/0.95',
        'reduce_poor_exposure_0.05': '1_reduce_total_exposure/q1/0.95',
        'reduce_total_vulnerability_0.05': '3_reduce_total_vulnerability/q1+q2+q3+q4+q5/0.95',
        'reduce_poor_vulnerability_0.05': '3_reduce_total_vulnerability/q1/0.95',
        'increase_gdp_pc_and_liquidity_0.05': '5_scale_income_and_liquidity/q1+q2+q3+q4+q5/1.05',
        'reduce_self_employment_0.1': '6_scale_self_employment/q1+q2+q3+q4+q5/0.9',
        'reduce_non_diversified_income_0.1': '7_scale_non_diversified_income/q1+q2+q3+q4+q5/0.9',
        'pds40': '8_post_disaster_support_imperfect/q1+q2+q3+q4+q5/0.4',
        # 'pds50': '8_post_disaster_support_imperfect/q1+q2+q3+q4+q5/0.5',
        # 'pds20_perfect': '8a_post_disaster_support_perfect/q1+q2+q3+q4+q5/0.2',
        'insurance20': '9_insurance/q1+q2+q3+q4+q5/0.2',
        'noLiquidity': '10_scale_income_and_liquidity/q1+q2+q3+q4+q5/0',
        'reduce_gini_10': '11_scale_gini_index/0.9',
    }
    simulation_paths = {k: os.path.join(args.simulation_outputs_dir, v) for k, v in simulation_paths.items()}

    income_groups, gini_index, cat_info_data, macro_data, results_data, name_dict, any_to_wb, iso3_to_wb, iso2_iso3 = load_data(simulation_paths, model_root_dir)

    gadm_world = gpd.read_file("/Users/robin/data/GADM/gadm_410-levels.gpkg", layer='ADM_0').set_crs(4326).to_crs('World_Robinson')
    gadm_world = gadm_world[~gadm_world.COUNTRY.isin(['Antarctica', 'Caspian Sea'])]
    gadm_world.rename(columns={'GID_0': 'iso3'}, inplace=True)

    if args.plot:
        plot_fig_1(
            cat_info_data_=cat_info_data['baseline'],
            macro_data_=macro_data['baseline'],
            countries=['HTI', 'TJK'],
            hazard='Earthquake',
            plot_rp=100,
            outpath=outpath,
        )

        plot_fig_2(
            data_=results_data['baseline'],
            world_=gadm_world,
            bins_list={'resilience': [10, 20, 30, 40, 50, 60, 70, 80], 'risk': [0, .25, .5, 1, 2, 6],
                       'risk_to_assets': [0, .125, .25, .5, 1, 3]},
            cmap={'resilience': 'Reds_r', 'risk': 'Reds', 'risk_to_assets': 'Reds'},
            annotate=['HTI', 'TJK'],
            outfile=f"{outpath}/fig_2.pdf",
            log_xaxis=True,
            run_ols=True,
            show=True,
        )

        plot_fig_3(
            results_data_=results_data['baseline'],
            cat_info_data_=cat_info_data['baseline'],
            outfile=f"{outpath}/fig_3.pdf",
            numbering=True,
            show=True,
        )

        plot_fig_4(
            cat_info_data_=cat_info_data['baseline'],
            income_groups_=income_groups,
            # map_bins=[0, .5, 1, 2, 4, 7],
            # map_bins=[1, 2.5, 5, 10, 20, 30],
            map_bins=[.5, 1, 2, 4, 8, 16, 30],
            plot_rp=None,
            world_=gadm_world,
            outfile=f"{outpath}/fig_4.pdf",
            show=True,
        )

        plot_fig_5(
            results_data_=results_data,
            cat_info_data_=cat_info_data,
            plot_rp=None,
            outfile=f"{outpath}/fig_5.pdf",
        )

        plot_supfig_2(
            cat_info_data_=cat_info_data['baseline'],
            macro_data_=macro_data['baseline'],
            iso3='HTI',
            hazard='Earthquake',
            plot_rp=100,
            show=True,
            outfile=f"{outpath}/supfig_2.pdf",
        )

        plot_supfigs_3_4(
            results_data_=results_data['baseline'],
            outpath_=outpath,
        )

        plot_supfig_5(
            results_data_=results_data['baseline'],
            outfile=f"{outpath}/supfig_5.pdf",
            show=True,
        )

        plot_supfig_6(
            results_data_=results_data['baseline'],
            outfile=f"{outpath}/supfig_6.pdf",
            show=True,
        )

        plot_supfig_7(
            results_data_=results_data['baseline'],
            cat_info_data_=cat_info_data['baseline'],
            plot_rp=None,
            outfile=f"{outpath}/supfig_7.pdf",
            show=True,
        )

        plot_supfig_8(
            cat_info_data_=cat_info_data['noLiquidity'],
            outfile=f"{outpath}/supfig_8.pdf",
            show=True,
            numbering=False,
            plot_rp=None,
        )

    print_stats(results_data['baseline'])

    print_results_table(results_data['baseline'])
