"""
  Copyright (c) 2023-2025 Robin Middelanis <rmiddelanis@worldbank.org>

  This file is part of the global Unbreakable model.

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
"""
import copy
import glob
import itertools
import shutil
import sys
import os
from pathlib import Path

import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.transforms import blended_transform_factory
import numpy as np
import pandas as pd
from misc.helpers import average_over_rp, get_population_scope_indices, calculate_average_recovery_duration
from post_processing.post_processing import preprocess_simulation_data
import seaborn as sns
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
import matplotlib.patches as patches
from matplotlib.colors import to_rgb
import matplotlib.ticker as mtick

# Set the default font size for labels and ticks
plt.rcParams['axes.labelsize'] = 7  # Font size for x and y labels
plt.rcParams['xtick.labelsize'] = 6  # Font size for x tick labels
plt.rcParams['ytick.labelsize'] = 6  # Font size for y tick labels
plt.rcParams['legend.fontsize'] = 6  # Font size for legend
plt.rcParams['axes.titlesize'] = 8  # Font size for axis title
plt.rcParams['font.size'] = 7  # Font size for text

# figure widths
# "Column-and-a-Half: 120–136 mm wide"
inch = 2.54
centimeter = 1 / inch

double_col_width = 19.4 * centimeter # cm
single_col_width = 9.5 * centimeter  # cm
max_fig_height = 24.7 * centimeter  # cm

WORLD_REGION_NAMES = {
    'ECA': 'Europe and Central Asia',
    'EAP': 'East Asia and Pacific',
    'LAC': 'Latin America and Caribbean',
    'MNA': 'Middle East and North Africa',
    'NAM': 'North America',
    'SAR': 'South Asia',
    'SSA': 'Sub-Saharan Africa',
}


def format_dollar_value(val, abbreviations=True):
    factor = 1
    unit_multiplier = ''
    if val > 1e3 and val < 1e6:
        factor = 1e3
        if abbreviations:
            unit_multiplier = "k"
        else:
            unit_multiplier = "thousands"
    elif val >= 1e6 and val < 1e9:
        factor = 1e6
        if abbreviations:
            unit_multiplier = "mn"
        else:
            unit_multiplier = "millions"
    elif val >= 1e9:
        factor = 1e9
        if abbreviations:
            unit_multiplier = "bn"
        else:
            unit_multiplier = "billions"
    decimals = 0
    if val / factor < 1000:
        decimals = 1
    elif val / factor < 100:
        decimals = 2
    elif val / factor < 10:
        decimals = 3
    elif val / factor < 1:
        decimals = 4
    return factor, unit_multiplier, decimals


def get_policy_name_lookup(df_):
    policy_names = {
        'insurance': 'Insurance covering ++vs++% of losses for for ++hs++',
        'post_disaster_support': 'Post-disaster support equal to ++vs++% of the losses of the poor for for ++hs++',
        'reduce_exposure': '++reduce_increase++ exposure by ++vs++% for for ++hs++',
        'reduce_vulnerability': '++reduce_increase++ vulnerability by ++vs++% for for ++hs++',
        'scale_non_diversified_income': '++reduce_increase++ non-diversified income share by ++vs++% for for ++hs++',
        'baseline': 'No policy',
        'reduce_total_vulnerability': '++reduce_increase++ total vulnerability by ++vs++%, targeting ++hs++',
        'reduce_total_exposure': '++reduce_increase++ total exposure by ++vs++%, targeting ++hs++',
        'scale_gini_index': '++reduce_increase++ inequality by ++vs++%',
        'scale_income_and_liquidity': '++reduce_increase++ income and liquidity by ++vs++% for for ++hs++',
        'scale_liquidity': '++reduce_increase++ liquidity by ++vs++% for for ++hs++',
        'scale_self_employment': '++reduce_increase++ self-employment rate by ++vs++%',
    }
    hs_lookup = {
        '20': 'the poorest 20% of the population',
        '40': 'the bottom 40% of the population',
        '60': 'the bottom 60% of the population',
        '80': 'the bottom 80% of the population',
        '100': 'the entire population',
    }
    if isinstance(df_, pd.DataFrame):
        policies = df_.index.get_level_values('policy').unique()
    elif isinstance(df_, (xr.DataArray, xr.Dataset)):
        policies = df_.policy.values
    else:
        raise ValueError("Input must be a pandas DataFrame or xarray DataArray.")
    policy_name_lookup_ = pd.Series(index=policies, dtype=str, name='policy_name')
    policy_name_lookup_.index.name = 'policy'
    for p in policies:
        policy_name_lookup_.loc[p] = policy_names[p.split('/')[0]].replace('++reduce_increase++', 'reduce' if p.split('/')[2][0] == '-' else 'increase').replace('++vs++', p.split('/')[2][1:]).replace('++hs++', hs_lookup.get(p.split('/')[1], ''))
    return policy_name_lookup_


def plot_fig_1(macro_res_, variables, country=None, outpath=None):
    if 'hs' in macro_res_.coords:
        baseline_results = macro_res_.loc[dict(policy='baseline', hs=0, vs=0)].to_dataframe()
        baseline_results = baseline_results.drop(columns=['policy', 'hs', 'vs']).dropna(how='all')
    else:
        baseline_results = macro_res_.loc[dict(policy='baseline/0/+0')].to_dataframe()
        baseline_results = baseline_results.drop(columns=['policy']).dropna(how='all')

    ncols = len(variables)
    fig, axs = plt.subplots(1, ncols, figsize=(double_col_width, 3 * centimeter))

    units = {
        'risk_to_assets': '%',
        'risk_to_consumption': '%',
        'risk_to_wellbeing': '%',
        't_reco_95': ' years',
        'risk_to_all_poverty': '%',
        'risk_to_societal_poverty': '%',
        'risk_to_extreme_poverty': '%',
        'resilience': '%',
    }
    prefixes = {
        'default': '',
        't_reco_95': '\n'
    }
    ax_titles = {
        'resilience': 'Socio-economic\nresilience [%]',
        'risk': 'Risk to\nwell-being [% GDP]',
        'risk_to_wellbeing': 'Risk to\nwell-being [% GDP]',
        'risk_to_consumption': 'Risk to\nconsumption [% GDP]',
        'risk_to_assets': 'Risk to\nassets [% GDP]',
        't_reco_95': 'Average recovery\ntime [years]',
        'risk_to_extreme_poverty': 'Risk to extreme\npoverty [% population]',
        'risk_to_societal_poverty': 'Risk to societal\npoverty [% population]',
        'risk_to_all_poverty': 'Risk to all\npoverty [% population]',
    }

    percent_units = [u for u in units.keys() if units[u] == '%']
    baseline_results.loc[:, np.intersect1d(percent_units, baseline_results.columns)] *= 100

    legend=True
    for ax, var in zip(axs, variables):
        ax.axhline(0, color='black', lw=0.5, alpha=.5)
        sns.scatterplot(data=baseline_results, x=var, y=0, ax=ax, color='lightgrey', edgecolor='none', alpha=0.1, s=100,
                        label='all countries', legend=legend)
        sns.scatterplot(x=[baseline_results[var].median()], y=[0], ax=ax, edgecolor='k', facecolor='none', s=100,
                        label='global median', legend=legend)
        sns.scatterplot(
            x=[baseline_results[baseline_results.region == baseline_results.loc[country, 'region']][var].median()],
            y=[0], ax=ax, edgecolor='k', facecolor='none', s=100, label='region median',
            linestyle='--', legend=legend, alpha=.5)
        if country is not None:
            sns.scatterplot(data=baseline_results.loc[[country]], x=var, y=0, ax=ax, edgecolor='darkturquoise',
                            facecolor='none', s=100, label=baseline_results.loc[country, 'name'], legend=legend,
                            zorder=10)
            ctry_val = baseline_results.loc[country, var]
            abs_val = ''
            if var not in ['t_reco_95', 'resilience']:
                abs_val_unit = ''
                if var in ['risk_to_assets', 'risk_to_consumption', 'risk_to_wellbeing']:
                    abs_val = baseline_results.loc[country, ['pop', 'gdp_pc_pp']].prod() * ctry_val / 100
                    abs_val_unit = '$'
                elif var in ['risk_to_all_poverty', 'risk_to_extreme_poverty', 'risk_to_societal_poverty']:
                    abs_val = baseline_results.loc[country, 'pop'] * ctry_val / 100
                    abs_val_unit = ''

                dollar_factor, dollar_unit_multiplier, abs_decimals = format_dollar_value(abs_val)
                abs_val = f"\n({abs_val_unit}{abs_val / dollar_factor:.{abs_decimals}f}{dollar_unit_multiplier})"

            rel_decimals = 2 if var in ['risk_to_assets', 'risk_to_consumption', 'risk_to_wellbeing'] else 1
            ax.text(ctry_val, 0.02, f"{prefixes.get(var, prefixes['default'])}{round(baseline_results.loc[country, var], rel_decimals)}{units[var]}" + abs_val, ha='center',
                    va='bottom', fontsize=7, color='darkturquoise')
        ax.text(baseline_results[var].median(), -.02, f"{np.round(baseline_results[var].median(), 2)}{units[var]}", ha='center',
                va='top', fontsize=7, color='k')
        ax.text(.5, 1, f"{ax_titles.get(var, var)}", fontsize=8, ha='center', va='bottom', transform=ax.transAxes)
        ax.set_xlim(min(ax.get_xlim()[0], -.1 * (baseline_results[var].max() - baseline_results[var].min())), ax.get_xlim()[1])
        ax.axis('off')
        legend=False

    handles, labels = axs[0].get_legend_handles_labels()
    handles[0] = copy.copy(handles[0])
    handles[0].set_alpha(.5)
    axs[0].legend(handles, labels, loc='center right', fontsize=7, frameon=False, bbox_to_anchor=(-.2, .5))
    plt.tight_layout()
    plt.draw()

    for ax, var in zip(axs, variables):
        # make sure markers are not cut off
        radius_pts = (100 / np.pi) ** 0.5  # Radius in points
        radius_pixels = radius_pts * fig.dpi / 72  # Radius in pixels
        inv = ax.transData.inverted()
        x0_data, _ = inv.transform((0, 0))
        x1_data, _ = inv.transform((radius_pixels, radius_pixels))

        # Calculate padding in data units
        x_pad = x1_data - x0_data

        # Set limits so the marker is fully shown
        ax.set_xlim(-x_pad, baseline_results[var].max() + x_pad)
        ax.set_ylim(-.036, .06)

    if outpath is not None:
        plt.savefig(outpath, dpi=300, bbox_inches='tight', transparent=True)

    plt.show(block=False)


def plot_fig_2(macro_inputs_, model_results_, t_reco_, country, outpath=None):
    fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(double_col_width, 6.1 * centimeter))

    plot_data = model_results_.loc[pd.IndexSlice['baseline/0/+0', country, 'all hazards', 'annual average', :]]
    plot_data = plot_data[['dk_pc', 'dc_pc', 'dw_pc_currency', 'resilience', 'n']]
    plot_data[['dk_tot', 'dc_tot', 'dw_tot']] = plot_data[['dk_pc', 'dc_pc', 'dw_pc_currency']].mul(plot_data.n, axis=0) * macro_inputs_.loc[('baseline/0/+0', country), 'pop']
    plot_data['resilience'] *= 100

    plot_data = pd.concat([plot_data, t_reco_.loc[pd.IndexSlice['baseline/0/+0', country, 'all hazards', :]].t_reco_avg],
                         axis=1)

    dollar_factor, dollar_unit_multiplier, _ = format_dollar_value(plot_data.dk_tot.min())
    plot_data[['dk_tot', 'dc_tot', 'dw_tot']] /= dollar_factor

    loss_data = plot_data.drop('total')[['dk_pc', 'dc_pc', 'dw_pc_currency']].rename(columns={'dk_pc': 'assets', 'dc_pc': 'consumption', 'dw_pc_currency': 'well-being'}).stack().rename('loss')
    loss_data.index.names = ['income_cat', 'loss_type']
    losses_color_palette = [tuple((1 - amount) * channel + amount for channel in to_rgb('darkturquoise')) for amount in [0, 0.3, .6]]
    sns.barplot(data=loss_data.to_frame(), x='income_cat', y='loss', hue='loss_type', ax=axs[0], palette=losses_color_palette,
                dodge=True, width=0.4)
    # Secondary axis: % of GDP
    twinx0 = axs[0].twinx()
    twinx0.set_ylabel('[% avg. per capita income]')
    # Set the secondary y-axis limits to match the % of GDP range
    gdp_pc = macro_inputs_.loc[('baseline/0/+0', country), 'gdp_pc_pp']
    twinx0.plot('q1', axs[0].get_ylim()[1] / gdp_pc * 100, alpha=0)
    twinx0.set_ylim((axs[0].get_ylim()[0] / gdp_pc * 100, (axs[0].get_ylim()[1] / gdp_pc * 100)))
    twinx0.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100))
    axs[0].set_title('Annual average losses per capita\n', ha='center', va='bottom', fontsize=8)
    axs[0].set_xlabel('Household income quintile')
    axs[0].set_ylabel(f'[$PPP]')
    axs[0].legend(loc='lower center', bbox_to_anchor=(0.5, .98), ncol=3, frameon=False, title=None, handlelength=1,
                  handletextpad=.5, labelspacing=0.)

    sns.barplot(plot_data.drop('total'), x='income_cat', y='resilience', ax=axs[1], color='darkturquoise', dodge=True,
                width=0.6, label='quintiles')
    axs[1].axhline(plot_data.loc['total', 'resilience'], color='darkturquoise', lw=0.5, ls='--', label='national average')
    axs[1].set_ylabel(None)
    # axs[1].legend(ncol=1, frameon=False, title=None, handlelength=1, handletextpad=.5)
    axs[1].legend(loc='lower center', bbox_to_anchor=(0.5, .98), ncol=3, frameon=False, title=None, handlelength=1,
                  handletextpad=.5, labelspacing=0.)
    axs[1].set_title('Socio-economic resilience [%]\n', ha='center', va='bottom', fontsize=8)
    axs[1].set_xlabel('Household income quintile')

    sns.barplot(plot_data.drop('total'), x='income_cat', y='t_reco_avg', ax=axs[2], color='darkturquoise', dodge=True,
                width=0.6, label='quintiles')
    axs[2].axhline(plot_data.loc['total', 't_reco_avg'], color='darkturquoise', lw=0.5, ls='--', label='national average')
    axs[2].set_ylabel(None)
    # axs[2].legend(ncol=1, frameon=False, title=None, handlelength=1, handletextpad=.5)
    axs[2].legend(loc='lower center', bbox_to_anchor=(0.5, .98), ncol=3, frameon=False, title=None, handlelength=1,
                  handletextpad=.5, labelspacing=0.)
    axs[2].set_title('Average recovery time [years]\n', ha='center', va='bottom', fontsize=8)
    axs[2].set_xlabel('Household income quintile')

    plt.tight_layout(w_pad=2)

    if outpath is not None:
        plt.savefig(outpath, dpi=300, bbox_inches='tight', transparent=True, pad_inches=0.1)

    plt.show(block=False)


def plot_fig_3(macro_inputs_, model_results_, country, outpath=None):
    fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(double_col_width, 6.1 * centimeter))

    plot_data = model_results_.loc[pd.IndexSlice['baseline/0/+0', country, :, :, 'total', :]]
    plot_data.loc[:, ['dk_tot', 'dc_tot', 'dw_tot']] = (plot_data[['dk_pc', 'dc_pc', 'dw_pc_currency']] * macro_inputs_.loc[('baseline/0/+0', country), 'pop']).values
    dollar_factor, dollar_unit_multiplier, _ = format_dollar_value(plot_data.loc['all hazards'].drop('annual average').dw_tot.max())
    plot_data.loc[:, ['dk_tot', 'dc_tot', 'dw_tot']] = plot_data.loc[:, ['dk_tot', 'dc_tot', 'dw_tot']] / dollar_factor

    loss_data = plot_data.drop("annual average", level='rp').loc['all hazards']
    loss_data = loss_data[['dk_tot', 'dc_tot', 'dw_tot']].rename(columns={'dk_tot': 'assets', 'dc_tot': 'consumption', 'dw_tot': 'well-being'}).stack().rename('loss')
    loss_data.index.names = ['rp', 'loss_type']
    loss_data.index = loss_data.index.set_levels(loss_data.index.levels[0].astype(int), level=0)

    losses_color_palette = [tuple((1 - amount) * channel + amount for channel in to_rgb('darkturquoise')) for amount in [0, 0.3, .6]]
    sns.barplot(loss_data.to_frame(), x='rp', y='loss', hue='loss_type', ax=axs[0], palette=losses_color_palette,
                dodge=True, width=0.8, legend=False)
    twinx0 = axs[0].twinx()
    twinx0.set_ylabel('[% GDP]')
    # Set the secondary y-axis limits to match the % of GDP range
    gdp = macro_inputs_.loc[('baseline/0/+0', country), ['gdp_pc_pp', 'pop']].prod(axis=0)
    twinx0.plot(axs[0].get_xlim()[0], axs[0].get_ylim()[1] * dollar_factor / gdp * 100, alpha=0)
    twinx0.set_ylim((axs[0].get_ylim()[0] * dollar_factor / gdp * 100, (axs[0].get_ylim()[1] * dollar_factor / gdp * 100)))
    twinx0.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=0))
    axs[0].set_title('Expected losses by\nreturn period', ha='center', va='bottom')
    axs[0].set_xlabel('Return period [years]')
    axs[0].set_ylabel(f'[$PPP {dollar_unit_multiplier}]')

    rp_weights = pd.Series(index=loss_data.index.get_level_values('rp').unique(),
                           data=1 / loss_data.index.get_level_values('rp').unique() - np.concatenate([1 / loss_data.index.get_level_values('rp').unique()[1:], [0]]))
    rp_weights['annual average'] = 1
    aal_shares = plot_data[['dk_tot', 'dc_tot', 'dw_tot']].mul(rp_weights, axis=0) / plot_data.loc[('all hazards', 'annual average'), ['dk_tot', 'dc_tot', 'dw_tot']] * 100

    aal_rp_shares = aal_shares.loc['all hazards'].drop('annual average').rename(columns={'dk_tot': 'assets', 'dc_tot': 'consumption', 'dw_tot': 'well-being'}).stack().rename('loss')
    aal_rp_shares.index.names = ['rp', 'loss_type']
    aal_rp_shares.index = aal_rp_shares.index.set_levels(aal_rp_shares.index.levels[0].astype(int), level=0)
    sns.barplot(aal_rp_shares.to_frame(), x='rp', y='loss', hue='loss_type', ax=axs[1], dodge=True,
                width=0.8, palette=losses_color_palette, legend=False)
    axs[1].set_title('Share of annual average\nlosses by return period', ha='center', va='bottom')
    axs[1].set_xlabel('Return period [years]')
    axs[1].set_ylabel('[% AAL]')

    aal_hazard_shares = aal_shares.xs('annual average', level='rp').drop('all hazards').rename(columns={'dk_tot': 'assets', 'dc_tot': 'consumption', 'dw_tot': 'well-being'}).stack().rename('loss')
    aal_hazard_shares.index.names = ['hazard', 'loss_type']
    sns.barplot(aal_hazard_shares.to_frame(), x='hazard', y='loss', hue='loss_type', ax=axs[2], dodge=True,
                width=0.8, palette=losses_color_palette)
    axs[2].set_title('Share of annual average\nlosses by hazard', ha='center', va='bottom')
    axs[2].set_xlabel('Hazard')
    axs[2].set_ylabel('[% AAL]')
    axs[2].legend(frameon=False, title="Loss type", title_fontsize=7, handlelength=1,
                  handletextpad=.5, bbox_to_anchor=(1.05, 1), loc='upper left')

    for ax, rotation in zip(axs, [40, 40, 25]):
        ax.tick_params(axis='x', rotation=rotation)

    plt.tight_layout(w_pad=0)

    axs[1].set_position(axs[1].get_position().translated(.03, 0))

    if outpath is not None:
        plt.savefig(outpath, dpi=300, bbox_inches='tight', transparent=True, pad_inches=0.1)

    plt.show(block=False)


def plot_fig_2_old(ensemble_cat_info_, country=None, outpath=None):
    if 'hs' in ensemble_cat_info_.coords:
        baseline_data = ensemble_cat_info_.loc[dict(policy='baseline', hs=0, vs=0)].to_dataframe()
        baseline_data = baseline_data.drop(columns=['policy', 'hs', 'vs']).dropna(how='all')
    else:
        baseline_data = ensemble_cat_info_.loc[dict(policy='baseline/0/+0')].to_dataframe()
        baseline_data = baseline_data.drop(columns=['policy']).dropna(how='all')
    plot_data = baseline_data.xs('not_helped', level='helped_cat').xs('a', level='affected_cat')[['transfers', 'c']].dropna(how='all').droplevel(['hazard', 'rp']).drop_duplicates()

    fig, axs = plt.subplots(ncols=2, figsize=(7 * centimeter, 4 * centimeter), width_ratios=[.5, .6])

    transfers_spending = (plot_data.transfers * plot_data.c).groupby('iso3').sum() / plot_data.c.groupby('iso3').sum() * 100
    transfers_spending = transfers_spending.rename('transfers_spending')
    axs[0].axhline(0, color='black', lw=0.5, alpha=.5)
    sns.scatterplot(data=transfers_spending.to_frame(), x='transfers_spending', y=0, ax=axs[0], color='lightgrey', edgecolor='none', alpha=0.1, s=100)
    if country is not None:
        sns.scatterplot(data=transfers_spending.loc[[country]].to_frame(), x='transfers_spending', y=0, ax=axs[0],
                        edgecolor='darkturquoise', facecolor='none', s=100)
        ctry_val = transfers_spending.loc[country]

        axs[0].text(ctry_val, 0.02, f"{round(ctry_val, 2)}%", ha='center', va='bottom', fontsize=8,
                    color='darkturquoise')
    sns.scatterplot(x=[transfers_spending.median()], y=[0], ax=axs[0], edgecolor='k', facecolor='none', s=100)
    axs[0].text(transfers_spending.median(), -.02, f"{np.round(transfers_spending.median(), 2)}%", ha='center',
            va='top', fontsize=8, color='k')
    axs[0].set_title('Total transfer\namount [% GDP]', ha='center', va='bottom',
                transform=axs[0].transAxes)
    axs[0].set_xlim(min(axs[0].get_xlim()[0], -.1 * (transfers_spending.max() - transfers_spending.min())),
                axs[0].get_xlim()[1])
    axs[0].axis('off')
    plt.draw()

    # make sure markers are not cut off
    radius_pts = (100 / np.pi) ** 0.5  # Radius in points
    radius_pixels = radius_pts * fig.dpi / 72  # Radius in pixels
    inv = axs[0].transData.inverted()
    x0_data, _ = inv.transform((0, 0))
    x1_data, _ = inv.transform((radius_pixels, radius_pixels))

    # Calculate padding in data units
    x_pad = x1_data - x0_data

    # Set limits so the marker is fully shown
    axs[0].set_xlim(-x_pad, transfers_spending.max() + x_pad)
    axs[0].set_ylim(-.1, .1)

    transfers_distribution = (plot_data.transfers * plot_data.c).rename('transfers').to_frame()
    for q, scope in zip(['q1', 'q2', 'q3', 'q4', 'q5'], [(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1)]):
        scope_indices = get_population_scope_indices([scope], transfers_distribution)
        transfers_distribution.loc[pd.IndexSlice[:, scope_indices], 'income_cat'] = q
    transfers_distribution = transfers_distribution.reset_index('income_cat', drop=True).groupby(['iso3', 'income_cat']).mean()
    transfers_distribution = transfers_distribution / transfers_distribution.groupby('iso3').sum() * 100

    global_median = transfers_distribution.groupby('income_cat').median()
    global_median['iso3'] = 'Global median'
    global_median = global_median.reset_index().set_index(['iso3', 'income_cat'])

    ctry_global = pd.concat([transfers_distribution.loc[[country]], global_median], axis=0)
    sns.barplot(data=ctry_global.reset_index(), x='income_cat', y='transfers', hue='iso3', ax=axs[1], palette=['darkturquoise', 'k'], dodge=True, width=0.4, alpha=0.75, legend=False)
    # sns.scatterplot(transfers_distribution, x='income_cat', y='transfers', ax=axs[1], color='lightgrey', edgecolor='none', alpha=0.1, s=100)
    # sns.scatterplot(transfers_distribution.groupby('income_cat').median(), x='income_cat',
    #                 y='transfers', ax=axs[1], edgecolor='k', facecolor='none', s=100)
    # if country is not None:
    #     sns.scatterplot(data=transfers_distribution.loc[[country]], x='income_cat', y='transfers', ax=axs[1],
    #                     edgecolor='darkturquoise', facecolor='none', s=100)
    #     # ctry_vals = transfers_distribution.loc[country].values.flatten()
    #     # for x_pos, y_pos, text in zip(['q1', 'q2', 'q3', 'q4', 'q5'], ctry_vals, [f"{int(np.round(cv))}" for cv in ctry_vals]):
    #     #     axs[1].text(x_pos, y_pos, text, ha='center', va='center', fontsize=6, color='darkturquoise')
    axs[1].set_title('Share of total transfers\nreceived by each quintile [%]', ha='center', va='bottom',)
    axs[1].set_xlabel('Household income quintile')
    axs[1].set_ylabel(None)

    axs[1].set_xlim(-.5, 4.5)

    # deactivate right and top spines
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['top'].set_visible(False)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3)

    if outpath is not None:
        plt.savefig(outpath, dpi=300, bbox_inches='tight', transparent=True, pad_inches=0.1)


def plot_fig_5(macro_inputs_, country_, outpath=None):
    fig, axs = plt.subplots(ncols=5, nrows=2, figsize=(double_col_width, 5 * centimeter))

    plot_data = macro_inputs_.loc['baseline/0/+0'].copy()

    plot_data[['gini_index', 'self_employment', 'avg_prod_k', 'ew', 'home_ownership_rate', 'transfers_share_GDP', 'k_household_share', 'k_pub_share', 'region']] *= 100
    plot_data['k_priv_share'] = 100 - plot_data['k_pub_share'] - plot_data['k_household_share']

    plot_vars = {
        'gdp_pc_pp': ['GDP per capita', '$PPP'],
        'gini_index': ['Inequality\n(Gini index)', '%'],
        'self_employment': ['Self-employment\nrate', '%'],
        'avg_prod_k': ['Avgerage capital\nproductivity', '%'],
        'ew': ['Early warning\ncoverage', '%'],
        'home_ownership_rate': ['Home ownership\nrate', '%'],
        'transfers_share_GDP': ['Social protection and\nprivate remittances', '% of GDP'],
        'k_household_share': ['Capital share,\nhouseholds', '%'],
        'k_priv_share': ['Capital share,\nother private assets', '%'],
        'k_pub_share': ['Capital share,\npublic assets', '%'],
    }

    legend = True
    for ax, (var, (display_name, unit)) in zip(axs.flatten(), plot_vars.items()):
        ax.axhline(0, color='black', lw=0.5, alpha=.5)
        sns.scatterplot(data=plot_data, x=var, y=0, ax=ax, color='lightgrey', edgecolor='none', alpha=0.1,
                        s=100, legend=legend, label='all countries')
        sns.scatterplot(x=[plot_data[var].median()], y=[0], ax=ax, edgecolor='k', facecolor='none', s=100,
                        legend=legend, label='global median')
        sns.scatterplot(x=[plot_data[plot_data.region == plot_data.loc[country_, 'region']][var].median()], y=[0],
                        ax=ax, edgecolor='k', facecolor='none', s=100, alpha=.5, linestyle='--', label='region median',
                        legend=legend)
        sns.scatterplot(data=plot_data.loc[[country_]], x=var, y=0, ax=ax, edgecolor='darkturquoise',
                        facecolor='none', s=100, legend=legend, label=plot_data.loc[country_, 'name'])
        legend = False

        ctry_val = plot_data.loc[country_, var]
        if unit == '$PPP':
            ax.text(ctry_val, .4, f"{unit} {ctry_val:,.0f}", ha='center', va='bottom', fontsize=6,
                    color='darkturquoise')
            ax.text(plot_data[var].median(), -.4, f"{unit} {plot_data[var].median():,.0f}", ha='center',
                    va='top', fontsize=6, color='k')
        else:
            ax.text(ctry_val, 0.4, f"{round(ctry_val, 1)}{unit}", ha='center', va='bottom', fontsize=6,
                    color='darkturquoise')
            ax.text(plot_data[var].median(), -.4, f"{int(np.round(plot_data[var].median(), 1))}%", ha='center',
                    va='top', fontsize=6, color='k')
        ax.set_title(display_name, ha='center', va='bottom', transform=ax.transAxes)
        ax.set_xlim(min(ax.get_xlim()[0], -.1 * (plot_data[var].max() - plot_data[var].min())), ax.get_xlim()[1])
        ax.axis('off')

        ax.set_ylim(-1, 1)

    handles, labels = axs.flatten()[0].get_legend_handles_labels()
    handles[0] = copy.copy(handles[0])
    handles[0].set_alpha(.5)
    axs.flatten()[0].legend(handles, labels, loc='upper right', fontsize=7, frameon=False, bbox_to_anchor=(-.1, 0.4),
                            title=None)
    plt.tight_layout(w_pad=3, h_pad=-1)
    plt.draw()

    for ax, var in zip(axs.flatten(), plot_vars.keys()):
        # make sure markers are not cut off
        radius_pts = (100 / np.pi) ** 0.5  # Radius in points
        radius_pixels = radius_pts * fig.dpi / 72  # Radius in pixels
        inv = ax.transData.inverted()
        x0_data, _ = inv.transform((0, 0))
        x1_data, _ = inv.transform((radius_pixels, radius_pixels))

        # Calculate padding in data units
        x_pad = x1_data - x0_data

        # Set limits so the marker is fully shown
        ax.set_xlim(-x_pad, plot_data[var].max() + x_pad)

    if outpath is not None:
        plt.savefig(outpath, dpi=300, bbox_inches='tight', transparent=True, pad_inches=0.1)


def plot_fig_6(cat_info_inputs_, macro_inputs_, country_, outpath=None):
    fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(double_col_width, 8 * centimeter))

    plot_data = cat_info_inputs_.loc[('baseline/0/+0', country_)].copy()
    plot_data['c_diversified'] = (plot_data['c'] * plot_data['diversified_share'])
    plot_data['c_labor'] = (plot_data['c'] * (1 - plot_data['diversified_share']))
    plot_data['k_hh'] = plot_data['k'] * macro_inputs_.loc[('baseline/0/+0', country_), 'k_household_share']
    plot_data['k_other'] = plot_data['k'] - plot_data['k_hh']
    plot_data['share_of_total_transfers'] = plot_data['transfers'] * plot_data['c'] / (plot_data['transfers'] * plot_data['c']).sum()
    plot_data = plot_data.rename(columns={'gamma_SP': 'share_of_total_diversified_income'})
    plot_data['share_of_total_diversified_income'] /= 5

    def add_percentage_twinx(ax_, total):
        twinx = ax_.twinx()
        twinx.plot('q1', total, alpha=0)
        twinx.set_ylim((ax_.get_ylim()[0] / total, (ax_.get_ylim()[1] / total)))
        twinx.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
        return twinx

    factor, unit, _ = format_dollar_value(plot_data.c.max(), abbreviations=False)
    unit = f", {unit}" if unit != '' else ''
    sns.barplot(data=plot_data[['c']] / factor, x='income_cat', y='c', ax=axs[0, 0], color='darkturquoise', dodge=True,
                width=0.6, label='diversified', alpha=0.5)
    sns.barplot(data=plot_data[['c_labor']] / factor, x='income_cat', y='c_labor', ax=axs[0, 0], color='darkturquoise',
                dodge=True, width=0.6, label='labor')
    axs[0, 0].set_ylabel(f"per capita\n[$PPP{unit}]")
    axs[0, 0].set_xlabel('')
    axs[0, 0].set_title('Income', ha='center', va='bottom')
    axs[0, 0].legend(loc='upper left', fontsize=6, frameon=False, title=None, handlelength=1, handletextpad=.5)
    twinx00 = add_percentage_twinx(axs[0, 0], plot_data['c'].sum() / factor)
    twinx00.set_ylabel('share of\ncountry total [%]')

    factor, unit, _ = format_dollar_value(plot_data.k.max(), abbreviations=False)
    unit = f", {unit}" if unit != '' else ''
    sns.barplot(data=plot_data[['k']] / factor, x='income_cat', y='k', ax=axs[0, 1], color='darkturquoise', dodge=True,
                width=0.6, label='hh-owned', alpha=0.5)
    sns.barplot(data=plot_data[['k_other']] / factor, x='income_cat', y='k_other', ax=axs[0, 1], color='darkturquoise',
                dodge=True, width=0.6, label='other')
    axs[0, 1].set_ylabel(f"per capita\n[$PPP{unit}]")
    axs[0, 1].set_xlabel('')
    axs[0, 1].set_title('Assets', ha='center', va='bottom')
    axs[0, 1].legend(loc='upper left', fontsize=6, frameon=False, title=None, handlelength=1, handletextpad=.5)
    twinx01 = add_percentage_twinx(axs[0, 1], plot_data['k'].sum() / factor)
    twinx01.set_ylabel('share of\ncountry total [%]')

    factor, unit, _ = format_dollar_value(plot_data.liquidity.max(), abbreviations=False)
    unit = f", {unit}" if unit != '' else ''
    sns.barplot(data=plot_data[['liquidity']] / factor, x='income_cat', y='liquidity', ax=axs[0, 2],
                color='darkturquoise', dodge=True, width=0.6)
    axs[0, 2].set_ylabel(f"per capita\n[$PPP{unit}]")
    axs[0, 2].set_xlabel('')
    axs[0, 2].set_title('Savings', ha='center', va='bottom')
    twinx02 = add_percentage_twinx(axs[0, 2], plot_data['liquidity'].sum() / factor)
    twinx02.set_ylabel('share of\ncountry total [%]')

    sns.barplot(data=plot_data[['axfin']], x='income_cat', y='axfin', ax=axs[1, 0], color='darkturquoise',
                dodge=True, width=0.6)
    axs[1, 0].set_ylabel(f"coverage [%]")
    axs[1, 0].set_xlabel(' ')
    axs[1, 0].set_title('Financial inclusion', ha='center', va='bottom')
    axs[1, 0].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))

    sns.barplot(data=plot_data[['diversified_share']], x='income_cat', y='diversified_share', ax=axs[1, 1],
                color='darkturquoise', dodge=True, width=0.6)
    axs[1, 1].set_ylabel(f"share of\nhousehold income [%]")
    axs[1, 1].set_xlabel(' ')
    axs[1, 1].set_title('Diversified income adequacy', ha='center', va='bottom')
    axs[1, 1].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))

    factor, unit, _ = format_dollar_value(plot_data.c_diversified.max(), abbreviations=False)
    unit = f", {unit}" if unit != '' else ''
    sns.barplot(data=plot_data[['c_diversified']] / factor, x='income_cat', y='c_diversified', ax=axs[1, 2],
                color='darkturquoise', dodge=True, width=0.6)
    axs[1, 2].set_ylabel(f"per capita\n[$PPP{unit}]")
    axs[1, 2].set_xlabel(' ')
    axs[1, 2].set_title('Diversified income distribution', ha='center', va='bottom')
    twinx10 = add_percentage_twinx(axs[1, 2], plot_data.c_diversified.sum() / factor)
    twinx10.set_ylabel('share of\ncountry total [%]')

    fig.text(0.5, 0, 'Household income quintile', fontsize=7, ha='center', va='bottom')

    plt.tight_layout(h_pad=2, w_pad=5)

    if outpath is not None:
        plt.savefig(outpath, dpi=300, bbox_inches='tight', transparent=True, pad_inches=0.1)


def plot_fig_7(hazard_inputs_quintile_, country, outpath=None):
    plot_data = hazard_inputs_quintile_.loc[('baseline/0/+0', country)].copy()

    hazards = list(plot_data.index.get_level_values('hazard').unique())
    if 'all hazards' in hazards:
        hazards.remove('all hazards')
        hazards = ['all hazards'] + hazards

    fig, axs = plt.subplots(ncols=len(hazards), nrows=2, sharex=True, sharey='row',
                             figsize=(double_col_width, 8 * centimeter))

    plot_protection_difference = True
    if (plot_data.xs(True, level='protection') == plot_data.xs(False, level='protection'))[['fa_avg', 'v_ew']].all().all():
        plot_protection_difference = False

    for i, hazard in enumerate(hazards):
        hazard_data = plot_data.loc[hazard].drop('total', level='income_cat')
        hazard_data_total = plot_data.loc[hazard].xs('total', level='income_cat')
        legend = plot_protection_difference and i == len(hazards) - 1
        if plot_protection_difference:
            sns.barplot(hazard_data.xs(False, level='protection'), x='income_cat', y='fa_avg', label='without',
                        ax=axs[0, i], color='darkturquoise', dodge=True, width=0.6, alpha=0.4, legend=legend)
        sns.barplot(hazard_data.xs(True, level='protection'), x='income_cat', y='fa_avg', label='with',
                    ax=axs[0, i], color='darkturquoise', dodge=True, width=0.6, legend=legend)
        if hazard == 'all hazards' and plot_protection_difference:
            helper_protected = hazard_data.v_ew * hazard_data.reset_index().protection.values
            helper_unprotected = hazard_data.v_ew * (~hazard_data.reset_index().protection).values
            sns.barplot(helper_protected.to_frame(), x='income_cat', y='v_ew', hue='protection', ax=axs[1, i],
                        palette=['darkturquoise', 'darkturquoise'], dodge=True, width=0.6, legend=False)
            sns.barplot(helper_unprotected.to_frame(), x='income_cat', y='v_ew', hue='protection', ax=axs[1, i],
                        palette=['darkturquoise', 'darkturquoise'], dodge=True, width=0.6, alpha=0.4, legend=False)
            axs[1, i].axhline(hazard_data_total.loc[False, 'v_ew'], color='darkturquoise', lw=0.5,
                              ls='--', label=None, alpha=0.4)
        else:
            sns.barplot(hazard_data.xs(True, level='protection'), x='income_cat', y='v_ew', ax=axs[1, i],
                        color='darkturquoise', dodge=True, width=0.6, legend=False)
        axs[1, i].axhline(hazard_data_total.loc[True, 'v_ew'], color='darkturquoise', lw=0.5,
                          ls='--', label=None)
        axs[0, i].set_title(hazard, ha='center', va='bottom')
        axs[1, i].set_xlabel(' ')

    axs[0, 0].set_ylabel('Annual average exposure\n[% total assets]')
    axs[1, 0].set_ylabel('Asset vulnerability [%]')

    fa_ymax = axs[0, 0].get_ylim()[1]
    fa_decimals = 1
    if fa_ymax < .1:
        fa_decimals = 2
    elif fa_ymax < .01:
        fa_decimals = 3
    elif fa_ymax < .001:
        fa_decimals = 4
    axs[0, 0].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=fa_decimals))
    axs[1, 0].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))

    if plot_protection_difference:
        axs[0, -1].legend(loc='upper left', fontsize=6, frameon=False, title="hazard protection", bbox_to_anchor=(1, 1))

    plt.tight_layout()

    rightmost_ax_edge = fig.transFigure.inverted().transform(axs[0, -1].transData.transform((axs[0, -1].get_xlim()[1], 0)))[0]
    fig.text((axs[0, 0].get_position().x0 + rightmost_ax_edge) / 2, 0.03, 'Household income quintile',
             fontsize=7, ha='center', va='bottom')

    if outpath is not None:
        plt.savefig(outpath, dpi=300, bbox_inches='tight', transparent=True, pad_inches=0.1)




def plot_fig_3_old(ensemble_results_, ensemble_cat_info_, country=None, outpath=None):
    baseline_selector = dict(policy='baseline/0/+0')
    if 'hs' in ensemble_cat_info_.coords:
        baseline_selector = dict(policy='baseline', hs=0, vs=0)

    fig, axs = plt.subplots(ncols=4, figsize=(11 * centimeter, 4 * centimeter))
    plot_data = pd.merge(
        ensemble_cat_info_.loc[baseline_selector].loc[dict(helped_cat='not_helped', affected_cat='a')][['c', 'diversified_share', 'liquidity', 'axfin', 'k']].to_dataframe()[['c', 'diversified_share', 'liquidity', 'axfin', 'k']].dropna(how='all').droplevel(['hazard', 'rp']).drop_duplicates(),
        ensemble_results_.loc[baseline_selector][['k_household_share']].to_dataframe().dropna(how='all')[['k_household_share']].drop_duplicates(),
        left_index=True, right_index=True, how='inner')
    plot_data['c_diversified'] = plot_data['c'] * plot_data['diversified_share']
    plot_data['c_labor'] = plot_data['c'] * (1 - plot_data['diversified_share'])
    plot_data['k_hh'] = plot_data['k'] * plot_data['k_household_share']
    plot_data['k_other'] = plot_data['k'] * (1 - plot_data['k_household_share'])
    plot_data[['c', 'c_diversified', 'c_labor', 'k', 'k_hh', 'k_other']] /= 1000
    plot_data['axfin'] *= 100
    plot_data = plot_data.loc[[country], ['c', 'c_diversified', 'c_labor', 'k', 'k_hh', 'k_other', 'liquidity', 'axfin']].dropna(how='all').drop_duplicates()

    for q, scope in zip(['q1', 'q2', 'q3', 'q4', 'q5'], [(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1)]):
        scope_indices = get_population_scope_indices([scope], plot_data)
        plot_data.loc[pd.IndexSlice[:, scope_indices], 'income_cat'] = q
    plot_data = plot_data.reset_index('income_cat', drop=True).groupby('income_cat').mean()

    sns.barplot(data=plot_data, x='income_cat', y='k', ax=axs[0], color='darkturquoise', dodge=True, width=0.6,
                label='hh-owned', alpha=0.5)
    sns.barplot(data=plot_data, x='income_cat', y='k_other', ax=axs[0], color='darkturquoise', dodge=True, width=0.6,
                label='other')
    axs[0].set_xlabel(None)
    axs[0].set_ylabel(None)
    axs[0].set_title('Assets\n[$PPP 1,000]', ha='center', va='bottom')
    axs[0].set_xlabel(' ')
    axs[0].legend(loc='upper left', fontsize=6, frameon=False, title=None, handlelength=1, handletextpad=.5)

    sns.barplot(data=plot_data, x='income_cat', y='c', ax=axs[1], color='darkturquoise', dodge=True, width=0.6,
                label='diversified', alpha=0.5)
    sns.barplot(data=plot_data, x='income_cat', y='c_labor', ax=axs[1], color='darkturquoise', dodge=True, width=0.6,
                label='labor')
    axs[1].set_ylabel(None)
    axs[1].set_title('Income\n[$PPP 1,000]', ha='center', va='bottom')
    axs[1].set_xlabel(None)
    axs[1].legend(loc='upper left', fontsize=6, frameon=False, title=None, handlelength=1, handletextpad=.5)

    liquidity_unit = '[$PPP]'
    if plot_data.liquidity.max() > 1e3:
        liquidity_unit = '[$PPP 1,000]'
        plot_data['liquidity'] /= 1000
    sns.barplot(data=plot_data, x='income_cat', y='liquidity', ax=axs[2], color='darkturquoise', dodge=True, width=0.6)
    axs[2].set_ylabel(None)
    axs[2].set_title(f'Savings\n{liquidity_unit}', ha='center', va='bottom')
    axs[2].set_xlabel(None)

    sns.barplot(data=plot_data, x='income_cat', y='axfin', ax=axs[3], color='darkturquoise', dodge=True, width=0.6)
    axs[3].set_ylabel(None)
    axs[3].set_title('Financial\ninclusion [%]', ha='center', va='bottom')
    axs[3].set_xlabel(None)

    fig.text(0.5, 0, 'Household income quintile', fontsize=7, ha='center', va='bottom', transform=fig.transFigure)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3)

    if outpath is not None:
        plt.savefig(outpath, dpi=300, bbox_inches='tight', transparent=True, pad_inches=0.1)


#
# def plot_fig_4a_old(ensemble_results_, country=None, outpath=None):
#     fig, ax = plt.subplots(figsize=(3 * centimeter, 4 * centimeter))
#
#     plot_data = ensemble_results_.loc[dict(policy='baseline', vs=0, hs=0)][['ew']].to_dataframe()['ew'].dropna(how='all') * 100
#     ax.axhline(0, color='black', lw=0.5, alpha=.5)
#     sns.scatterplot(data=plot_data.to_frame(), x='ew', y=0, ax=ax, color='lightgrey', edgecolor='none', alpha=0.1, s=100)
#     if country is not None:
#         sns.scatterplot(data=plot_data.loc[[country]].to_frame(), x='ew', y=0, ax=ax, edgecolor='darkturquoise',
#                         facecolor='none', s=100)
#         ctry_val = plot_data.loc[country]
#
#         ax.text(ctry_val, 0.02, f"{round(ctry_val, 2)}%", ha='center', va='bottom', fontsize=8,
#                     color='darkturquoise')
#     sns.scatterplot(x=[plot_data.median()], y=[0], ax=ax, edgecolor='k', facecolor='none', s=100)
#     ax.text(plot_data.median(), -.02, f"{int(np.round(plot_data.median(), 0))}%", ha='center',
#             va='top', fontsize=8, color='k')
#     ax.set_title('Access to\nearly warning [%]', fontsize=7, ha='center', va='bottom',
#                 transform=ax.transAxes)
#     ax.set_xlim(min(ax.get_xlim()[0], -.1 * (plot_data.max() - plot_data.min())),
#                 ax.get_xlim()[1])
#     ax.axis('off')
#     plt.draw()
#
#     # make sure markers are not cut off
#     radius_pts = (100 / np.pi) ** 0.5  # Radius in points
#     radius_pixels = radius_pts * fig.dpi / 72  # Radius in pixels
#     inv = ax.transData.inverted()
#     x0_data, _ = inv.transform((0, 0))
#     x1_data, _ = inv.transform((radius_pixels, radius_pixels))
#
#     # Calculate padding in data units
#     x_pad = x1_data - x0_data
#
#     # Set limits so the marker is fully shown
#     ax.set_xlim(-x_pad, plot_data.max() + x_pad)
#     ax.set_ylim(-.1, .1)
#
#     plt.tight_layout()
#
#     if outpath is not None:
#         plt.savefig(outpath, dpi=300, bbox_inches='tight', transparent=True, pad_inches=0.1)
#
#
# def plot_fig_4b_old(ensemble_cat_info_, country=None, outpath=None):
#     plot_data = ensemble_cat_info_.loc[dict(policy='baseline', vs=0, hs=0, iso3=country)][['fa', 'v_ew']].to_dataframe()[['fa', 'v_ew']].dropna(how='all') * 100
#     plot_data = pd.concat([
#         plot_data.v_ew.loc[pd.IndexSlice[:, 10, :, 'a', 'not_helped']],
#         average_over_rp(plot_data.fa).loc[pd.IndexSlice[:, :, 'a', 'not_helped']]
#     ], axis=1).dropna(how='all')
#     hazards = plot_data.index.get_level_values('hazard').unique()
#     if 'Storm surge' in hazards and 'Wind' in hazards:
#         hazards = [h for h in hazards if h not in ['Storm surge', 'Wind']] + ['Storm surge', 'Wind']
#     num_hazards = len(hazards)
#
#     for q, scope in zip(['q1', 'q2', 'q3', 'q4', 'q5'], [(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1)]):
#         scope_indices = get_population_scope_indices([scope], plot_data)
#         plot_data.loc[pd.IndexSlice[:, scope_indices], 'income_cat'] = q
#     plot_data = plot_data.reset_index('income_cat', drop=True).groupby(['hazard', 'income_cat']).mean()
#
#     fig, axs = plt.subplots(figsize=(3 * num_hazards * centimeter, 7 * centimeter), ncols=num_hazards, nrows=2, sharex=True, sharey='row')
#
#     for axs_, var in zip(axs, ['fa', 'v_ew']):
#         for ax, hazard in zip(axs_, hazards):
#             sns.barplot(data=plot_data.xs(hazard, level='hazard')[[var]], x='income_cat', y=var, ax=ax, width=0.6,
#                         color='darkturquoise', dodge=True)
#             if var == 'fa':
#                 ax.set_title(hazard, fontsize=7, ha='center', va='bottom')
#             ax.set_xlabel(None)
#             ax.set_ylabel(None)
#             ax.set_xlabel(' ')
#
#     axs[0, 0].set_ylabel('Asset exposure [%]', fontsize=7)
#     axs[1, 0].set_ylabel('Asset vulnerability [%]', fontsize=7)
#
#     plt.tight_layout()
#     plt.subplots_adjust(wspace=0.2, hspace=0.2)
#
#     fig.text(0.5, 0., 'Household income quintile', fontsize=7, ha='center', va='bottom', transform=fig.transFigure)
#
#     if outpath is not None:
#         plt.savefig(outpath, dpi=300, bbox_inches='tight', transparent=True, pad_inches=0.1)
#

def get_optimal_trajectory(data, ax=None, interpolate_path=True, linestyle='solid'):
    if not interpolate_path:
        # find starting point
        num_steps = pd.DataFrame((np.ones(data.shape).cumsum(axis=0) + np.ones(data.shape).cumsum(axis=1) - 2))

        max_indices = num_steps.where(data.values == data.max().max())
        max_indices = max_indices.where(max_indices == max_indices.min().min()).stack().index

        def get_paths(idx_from, idx_to):
            i_, j_ = idx_from
            target_i_, target_j_ = idx_to
            possible_paths = list(set(list(itertools.permutations([(1, 0)] * (target_i_ - i_) + [(0, 1)] * (target_j_ - j_)))))
            possible_paths = [((0, 0),) + path for path in possible_paths]
            possible_paths = [[np.array(step) for step in path] for path in possible_paths]
            possible_paths = [[np.array([i_, j_]) + sum(path[:i + 1]) for i in range(len(path))] for path in possible_paths]
            possible_paths = [[tuple(step) for step in path] for path in possible_paths]
            path_values = [[data.iloc[i, j] for i, j in path] for path in possible_paths]
            cumulative_path_values = [[sum([data.iloc[i, j] for i, j in path[:i + 1]]) for i in range(len(path))] for path in possible_paths]
            possible_paths = pd.DataFrame(possible_paths)
            path_values = pd.DataFrame(path_values)
            cumulative_path_values = pd.DataFrame(cumulative_path_values)
            return possible_paths, path_values, cumulative_path_values

        plot_paths = []
        for target_idx in max_indices:
            paths, values, cum_values = get_paths((0, 0), target_idx)
            for step in paths.columns:
                keep_idx = values[values.iloc[:, step] == values.iloc[:, step].max()].index
                paths = paths.loc[keep_idx]
                values = values.loc[keep_idx]
                cum_values = cum_values.loc[keep_idx]
            sel_paths = paths[cum_values.iloc[:, -1] == cum_values.max().max()].dropna()
            sel_values = cum_values[cum_values.iloc[:, -1] == cum_values.max().max()].dropna()
            if len(sel_paths) > 1:
                simplified = pd.DataFrame(columns=sel_paths.columns)
                # for path_idx in sel_paths.index[:1]:
                for path_idx in sel_paths.index:
                    points, points_values = sel_paths.loc[path_idx], sel_values.loc[path_idx]
                    p_from_idx = 0
                    simplified_point_idx = 0
                    simplified.loc[path_idx, simplified_point_idx] = points.iloc[p_from_idx]
                    while p_from_idx < len(points) - 1:
                        p_to_idx = len(points) - 1
                        while p_to_idx > p_from_idx:
                            paths_to_from, _, values_to_from = get_paths(points.iloc[p_from_idx], points.iloc[p_to_idx])
                            has_angle = points.iloc[p_from_idx][0] != points.iloc[p_to_idx][0] and points.iloc[p_from_idx][1] != points.iloc[p_to_idx][1]
                            if ((values_to_from.duplicated().sum() == len(values_to_from) - 1) and has_angle) or p_to_idx == p_from_idx + 1:
                                simplified_point_idx += 1
                                simplified.loc[path_idx, simplified_point_idx] = points.iloc[p_to_idx]
                                p_from_idx = p_to_idx
                            else:
                                p_to_idx -= 1
                sel_paths = simplified.drop_duplicates()
                sel_values = sel_values.loc[sel_paths.index]
            plot_paths = plot_paths + [sel_paths.iloc[i].dropna().to_list() for i in range(len(sel_paths))]

        if ax is not None:
            # for segment in segments:
            for path in plot_paths:
                for i, segment in enumerate(zip(path[:-1], path[1:])):
                    y_, x_ = zip(segment[0], segment[1])
                    x_, y_ = np.array(x_) + .5, np.array(y_) + .5
                    linestyle = 'solid'
                    if np.sqrt(((x_[0] - x_[1]) ** 2 + (y_[0] - y_[1]) ** 2)) > 1:
                        linestyle = 'dotted'
                    ax.plot(x_, y_, linestyle=linestyle, color='grey', linewidth=1, markersize=6, marker='o')
                    if i == 0:
                        ax.plot(x_[0], y_[0], marker='s', color='grey', markersize=6)
                    elif i == len(path) - 2:
                        ax.plot(x_[1], y_[1], marker='s', color='grey', markersize=6)
                    dx, dy = x_[1] - x_[0], y_[1] - y_[0]
                    ax.arrow(x_[0] + 0.6 * dx, y_[0] + 0.6 * dy, .1 * dx, .1 * dy, head_width=0.08, head_length=0.08, fc='grey', ec='grey', length_includes_head=True)
        return plot_paths
    else:
        # interpolate
        x = np.arange(data.shape[1])
        y = np.arange(data.shape[0])
        z = data.values
        f = RegularGridInterpolator((y, x), z, method='pchip')
        xi = np.linspace(0, x[-1], 25)
        yi = np.linspace(0, y[-1], 25)
        xx, yy = np.meshgrid(xi, yi)
        zi = f((yy, xx))
        dy, dx = np.gradient(zi)
        dx_dy = np.sqrt(dx ** 2 + dy ** 2)

        # Function to get index on interpolated grid
        x_idx = lambda val: np.abs(xi - val).argmin()
        y_idx = lambda val: np.abs(yi - val).argmin()

        # Trace steepest ascent path from (0, 0)
        path_x, path_y = [0.], [0.]
        x_, y_ = 0., 0.
        value_ = zi[0, 0]
        # target = zi.max().max() * (1 - 1e-3)
        target = zi.max().max() - 0.1 * (sorted(z.flatten())[-1] - sorted(z.flatten())[-2])

        step_size = 0.01
        while value_ < target and (x_ + step_size <= x[-1] or y_ + step_size <= y[-1]):
            x_ = path_x[-1]
            y_ = path_y[-1]
            x_idx_ = x_idx(x_)
            y_idx_ = y_idx(y_)
            value_ = zi[y_idx_, x_idx_]
            dx_, dy_ = max(0, dx[y_idx_, x_idx_]), max(0, dy[y_idx_, x_idx_]) # only allow ascent

            if x_ + step_size > x[-1]:
                dx_ = 0
            if y_ + step_size > y[-1]:
                dy_ = 0

            # if gradient is zero (at start, because of grid bounds or due to local maximum), move towards the
            # closest positive allowed gradient
            if dx_ == 0 and dy_ == 0:
                distances = np.sqrt((xx - x_) ** 2 + (yy - y_) ** 2)
                masked_distances = np.where((dx_dy > 0) & (distances > 0), distances, np.inf) # allow positive gradients with increasing x and y
                masked_distances[:y_idx_, :] = np.inf # do not allow previous indices
                masked_distances[:, :x_idx_] = np.inf # do not allow previous indices
                masked_distances[y_idx_, x_idx_] = np.inf # do not allow current index
                masked_gradients = np.where(masked_distances == masked_distances.min(), dx_dy, 0)

                # average over all indices with the maximum gradient
                dy_ = np.mean([yy[i, j] - y_ for i, j in zip(*np.where(masked_gradients==masked_gradients.max()))])
                dx_ = np.mean([xx[i, j] - x_ for i, j in zip(*np.where(masked_gradients==masked_gradients.max()))])

                dy_ = max(0., dy_)
                dx_ = max(0., dx_)

                # if (masked_distances==masked_distances.min()).sum() > 1:
                #     print('breakpoint')
                #
                # min_index_flat = np.argmin(masked_distances)
                # min_index = np.unravel_index(min_index_flat, distances.shape)
                # dy_, dx_ = max(0, yy[min_index[0], min_index[1]] - y_), max(0, xx[min_index[0], min_index[1]] - x_)

                step = step_size / np.sqrt(dx_ ** 2 + dy_ ** 2)

                # check if we are still in the allowed grid
                if (not 0 <= x_ + dx_ * step <= x[-1]) and (not 0 <= y_ + dy_ * step <= y[-1]):
                    print("Reached bounds of grid without finding maximum.")
                    break
                elif not 0 <= x_ + dx_ * step <= x[-1]:
                    dx_ = 0
                elif not 0 <= y_ + dy_ * step <= y[-1]:
                    dy_ = 0
                if dx_ == 0 and dy_ == 0:
                    print("Could not find a positive gradient within grid bounds.")
                    break

            # recalculate step iff dx_ or dy_ were capped
            step = step_size / np.sqrt(dx_ ** 2 + dy_ ** 2)

            path_x.append(x_ + dx_ * step)
            path_y.append(y_ + dy_ * step)
            # print(path_x[-1], path_y[-1], dx_, dy_, value_)

        path_x = np.array(path_x) + .5
        path_y = np.array(path_y) + .5
        if ax is not None:
            ax.plot(path_x, path_y, linestyle=linestyle, color='grey', linewidth=2)
        return path_x, path_y


def plot_fig_4(macro_results_, country, metrics, scaling_selection=None, outpath=None):
    metric_names = {
        'risk_to_assets': 'Avoided risk to\nassets',
        'risk_to_consumption': 'Avoided risk to\nconsumption',
        'risk_to_wellbeing': 'Avoided risk to\nwell-being',
        'resilience': 'Socio-economic\nresilience increase',
        't_reco_95': 'Recovery time\nreduction',
        'risk_to_all_poverty': 'Avoided risk to total\npoverty',
        'risk_to_societal_poverty': 'Avoided risk to societal\npoverty',
        'risk_to_extreme_poverty': 'Avoided risk to extreme\npoverty',
    }

    decimals = {
        'risk_to_assets': 0,
        'risk_to_consumption': 0,
        'risk_to_wellbeing': 0,
        't_reco_95': 2,
        'resilience': 1,
        'risk_to_all_poverty': 0,
        'risk_to_societal_poverty': 0,
        'risk_to_extreme_poverty': 0,
    }

    policy_names = {
        'insurance/100/+20': 'Insurance covering\n20% of losses',
        'post_disaster_support/100/+40': 'Post-disaster support\nequal to 40% of the\nlosses of the poor',
        'reduce_total_exposure/100/-5': 'Reduce total exposure\nby 5% targeting the\nentire population',
        'reduce_total_exposure/20/-5': 'Reduce total exposure\nby 5% targeting the\npoorest 20%',
        'reduce_total_vulnerability/100/-5': 'Reduce total vulnerability\nby 5% targeting the\nentire population',
        'reduce_total_vulnerability/20/-5': 'Reduce total vulnerability\nby 5% targeting the\npoorest 20%',
        'scale_gini_index/100/-10': 'Reduce inequality by 10%',
        'scale_income_and_liquidity/100/+5': 'Increase income and\nliquidity by 5%',
        'scale_non_diversified_income/100/-10': 'Reduce non-diversified\nincome share by 10%',
        'scale_self_employment/100/-10': 'Reduce self-employment\nrate by 10%',
    }

    baseline_selector = dict(policy='baseline/0/+0')
    if 'hs' in macro_results_.coords:
        baseline_selector = dict(policy='baseline', hs=0, vs=0)
    else:
        scaling_selection = {p: {'policy': p} for p in macro_results_.policy.values if 'baseline' not in p}


    policies = macro_results_.policy.values
    dollar_metrics = np.intersect1d(['risk_to_assets', 'risk_to_consumption', 'risk_to_wellbeing'], metrics)
    population_metrics = np.intersect1d(['risk_to_all_poverty', 'risk_to_extreme_poverty', 'risk_to_societal_poverty'], metrics)
    plot_data = macro_results_[metrics]

    if scaling_selection is not None:
        mask = xr.zeros_like(plot_data, dtype=bool)
        mask.loc[baseline_selector] = True
        for policy, policy_dict in scaling_selection.items():
            mask.loc[policy_dict] = True
        plot_data = plot_data.where(mask)

    if len(population_metrics) > 0:
        plot_data[population_metrics] = plot_data[population_metrics] * macro_results_['pop']
    if len(dollar_metrics) > 0:
        plot_data[dollar_metrics] = plot_data[dollar_metrics] * macro_results_['gdp_pc_pp'] * macro_results_['pop']

    diff_abs = plot_data.sel(baseline_selector) - plot_data
    diff_abs = diff_abs.where((np.abs(diff_abs) >= 1e-10) | np.isnan(diff_abs), 0)

    diff_rel = diff_abs / plot_data.sel(baseline_selector) * 100
    diff_abs = diff_abs.drop_sel(baseline_selector)
    diff_rel = diff_rel.drop_sel(baseline_selector)

    if 'resilience' in plot_data:
        diff_abs['resilience'] *= -100
        diff_rel['resilience'] *= -1

    max_dollarval = xr.concat(diff_abs.sel(dict(iso3=country))[dollar_metrics].data_vars.values(), dim='var').max(dim='var').max().item()
    dollar_div, unit_multiplier, decimals = format_dollar_value(max_dollarval, abbreviations=True)
    diff_abs[dollar_metrics] = diff_abs[dollar_metrics] / dollar_div

    def get_value_with_unit(value, metric):
        if metric in dollar_metrics:
            return f"$PPP {value:.{decimals}f} {unit_multiplier}"
        elif metric in population_metrics:
            return f"{value:.0f} people"
        elif metric == 't_reco_95':
            return f"{value:.1f} years"
        elif metric == 'resilience':
            return f"{value:.1f} pp"
        else:
            raise ValueError(f"Unknown metric: {metric}")

    if scaling_selection is not None:
        fig_height = min(2 * (len(scaling_selection) - 1) + 2, 25) * centimeter
        fig, axs = plt.subplots(figsize=(double_col_width, fig_height), nrows=1, ncols=len(metrics) + 1, sharex='col',
                                width_ratios=[1.2] + [1] * len(metrics))
        axs[0].set_visible(False)
        # twin_axs = [ax.twiny() for ax in axs]
        for row_idx, (policy, policy_dict) in enumerate(scaling_selection.items()):
            for col_idx, (ax, metric) in enumerate(zip(axs[1:], metrics)):
                legend = row_idx == 0 and col_idx == len(metrics) - 1
                scenario_diff_rel = diff_rel.sel(policy_dict)[metric].to_dataframe()[[metric]]
                scenario_diff_abs = diff_abs.sel(policy_dict)[metric].to_dataframe()[[metric]]
                country_diff_rel = scenario_diff_rel.loc[country].item()
                sns.scatterplot(data=scenario_diff_rel, x=metric, y=row_idx, ax=ax, color='lightgrey', edgecolor='none',
                                alpha=0.05, s=100, legend=legend, label='all countries')
                sns.scatterplot(x=[scenario_diff_rel.median().item()], y=[row_idx], ax=ax, edgecolor='k',
                                facecolor='none', s=100, legend=legend, label='global median')
                region_countries = macro_res.where(macro_res.region == macro_res.sel(iso3=country).region.values[0],
                                                   drop=True).iso3.values
                sns.scatterplot(x=[scenario_diff_rel.loc[region_countries].median().item()], y=[row_idx], ax=ax,
                                edgecolor='k', facecolor='none', s=100, legend=legend, label='region median', alpha=.5,
                                linestyle='--')
                sns.scatterplot(x=[country_diff_rel], y=[row_idx], ax=ax, edgecolor='darkturquoise',
                                label=macro_results_.name.sel(baseline_selector).sel(iso3=country).item(),
                                legend=legend, facecolor='none', s=100, zorder=10)
                if legend:
                    handles, labels = ax.get_legend_handles_labels()

                country_diff_abs = scenario_diff_abs.loc[country].item()
                ax.text(country_diff_rel, row_idx + .15, f"{get_value_with_unit(country_diff_abs, metric)}",
                        ha='center', va='bottom', fontsize=7, color='darkturquoise')
                # twiny.plot(country_diff_abs, row_idx, alpha=0)

                if row_idx == len(scaling_selection) - 1:
                    ax.set_xlabel(f"{metric_names[metric]}[%]\n")

                    # ax.set_xlabel('relative change [%]')
                    ax.set_ylabel('')
                    ax.set_yticks([])
                    ax.set_yticklabels([])
                    ax.axvline(0, color='black', lw=0.5)
                    ax.grid(axis='x', linestyle='--', color='k', alpha=0.7)

                    # Remove all spines except the bottom one
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_visible(False)

        handles[0] = copy.copy(handles[0])
        handles[0].set_alpha(.5)
        axs[-1].legend(handles, labels, loc='lower right', fontsize=7, frameon=False, bbox_to_anchor=(1, 1.01))

        plt.tight_layout()
        plt.subplots_adjust(wspace=.25)
        plt.draw()

        for row_idx, (policy, policy_dict) in enumerate(scaling_selection.items()):
            for col_idx, (ax, metric) in enumerate(zip(axs[1:], metrics)):
                # make sure markers are not cut off
                radius_pts = (100 / np.pi) ** 0.5  # Radius in points
                radius_pixels = radius_pts * fig.dpi / 72  # Radius in pixels
                inv = ax.transData.inverted()
                x0_data, _ = inv.transform((0, 0))
                x1_data, _ = inv.transform((radius_pixels, radius_pixels))

                # Calculate padding in data units
                x_pad = x1_data - x0_data

                # Set limits so the marker is fully shown
                ax.set_xlim(min(-x_pad, ax.get_xlim()[0]), max(diff_rel.sel(policy_dict)[metric].to_dataframe()[[metric]].max().item() + x_pad, ax.get_xlim()[1]))

                if len(policies) > 1 and col_idx == 0:
                    policy_name = policy_names[policy]
                    ax.text(0, row_idx, policy_name, ha='left', va='center', wrap=True,
                            transform=blended_transform_factory(axs[0].transAxes, ax.transData))

                rect = patches.Rectangle((0, 0), 1, 1, transform=ax.transAxes, color='white', zorder=-1, clip_on=False)
                ax.add_patch(rect)
    else:
        fig_height = min(3.5 * (len(policies) - 1) + 2, 25) * centimeter
        fig, axs_ = plt.subplots(figsize=(19 * centimeter, fig_height), nrows=len(policies) - 1, ncols=len(metrics) + 1,
                                 width_ratios=[5] * len(metrics) + [.25])
        axs_ = axs_.reshape(len(policies) - 1, len(metrics) + 1)

        for row_idx, (axs, policy) in enumerate(zip(axs_, list(set(policies) - {'baseline'}))):
            for col_idx, (ax, metric) in enumerate(zip(axs, metrics)):
                cbar_ax = None
                if col_idx == 0:
                    cbar_ax = axs[-1]
                norm = mcolors.Normalize(vmin=0, vmax=xr.concat(diff_rel.data_vars.values(), dim='var').max(dim='var').max().item())
                annot_data = diff_abs[metric].sel(policy=policy, iso3=country).to_dataframe(name='z').z.unstack('vs').T.sort_index()
                annot_data= annot_data.dropna(axis=0, how='all').dropna(axis=1, how='all')
                heatmap_data = diff_rel[metric].sel(policy=policy, iso3=country).to_dataframe(name='z').z.unstack('hs')
                heatmap_data = heatmap_data.loc[annot_data.index, annot_data.columns].fillna(0)
                # annot_mask = pd.DataFrame(index=annot_data.index, columns=annot_data.columns, data=False)
                # if (annot_data==annot_data.max().max()).sum().sum() < 0.25 * np.prod(annot_data.shape):
                #     annot_mask = annot_data == annot_data.max().max()
                # else:
                #     annot_mask.iloc[-1, -1] = True
                # annot_mask.iloc[0, 0] = annot_mask.iloc[-1, 0] = annot_mask.iloc[0, -1] = True
                annot_data = annot_data.map(lambda x: f"{x:.{decimals[metric]}f}")
                # annot_data = annot_data.mask(~annot_mask).fillna('')
                if np.prod(heatmap_data.shape) == 0:
                    continue
                sns.heatmap(
                    heatmap_data, annot=annot_data, ax=ax, cbar_ax=cbar_ax, cmap='Blues', fmt='',
                    cbar_kws={'label': 'avoided impacts [%]'}, norm=norm, cbar=cbar_ax, annot_kws={"fontsize": 6})
                if not heatmap_data.isna().all().all():
                    get_optimal_trajectory(heatmap_data, ax)
                ax.set_ylim(0, len(heatmap_data))
                if row_idx == 0:
                    ax.set_title(f"{metric_names[metric]} [{get_value_with_unit(metric)}]")
                if row_idx != len(policies) - 2:
                    ax.set_xticklabels([])
                ax.set_xlabel(' \n')
                if col_idx == 0:
                    if len(policies) > 1:
                        ax.set_ylabel(policy_names[policy] + "\n\n", fontdict={'fontweight': 'bold', 'va': 'bottom'})
                    else:
                        ax.set_ylabel(" \n")
                else:
                    ax.set_ylabel(' \n')
                    ax.set_yticklabels([])

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.1, wspace=0.15)

        for ax in axs_[1:, -1]:
            ax.set_visible(False)
        cbar_height = min(9 * centimeter / fig_height, 1)
        cbar_x0 = axs_[0, -1].get_position().x0
        cbar_x1 = axs_[0, -1].get_position().x1
        cbar_y0 = axs_[-1, -1].get_position().y0 + 0.5 * (1 - cbar_height) * (axs_[0, -1].get_position().y1 - axs_[-1, -1].get_position().y0)
        cbar_y1 = axs_[0, -1].get_position().y1 - 0.5 * (1 - cbar_height) * (axs_[0, -1].get_position().y1 - axs_[-1, -1].get_position().y0)
        axs_[0, -1].set_position([cbar_x0, cbar_y0, cbar_x1 - cbar_x0, cbar_y1 - cbar_y0])

        fig.text(.037 if len(policies) > 1 else .013,
                 (axs_[0, 0].get_position().y1 + axs_[-1, 0].get_position().y0) / 2, f'vertical expansion [%]',
                 va='center', rotation='vertical', transform=fig.transFigure)
        fig.text(.5, 0.01, 'horizontal expansion\n[% population]', ha='center', va='bottom', transform=fig.transFigure)

    if outpath is not None:
        plt.savefig(outpath, dpi=300, transparent=True)

    plt.show(block=False)


def prepare_data(cat_info_res_, macro_res_, hazard_prot_sc_, cat_info_sc_, macro_sc_, hazard_ratios_sc_, capital_shares_, gini_index_, data_coverage_, excel_outpath=None):
    policy_name_lookup_ = get_policy_name_lookup(cat_info_res_)

    policy_dims = ['policy']
    if 'vs' in cat_info_res_.dims and 'hs' in cat_info_res_.dims:
        policy_dims += ['hs', 'vs']

    cat_info_res_df = cat_info_res_.to_dataframe()
    cat_info_sc_df_ = cat_info_sc_.to_dataframe()
    cat_info_sc_df_ = cat_info_sc_df_.rename(index={.2: 'q1', .4: 'q2', .6: 'q3', .8: 'q4', 1: 'q5'}, level='income_cat')
    cat_info_sc_df_['policy-iso3-income_cat'] = cat_info_sc_df_.reset_index()[['policy', 'iso3', 'income_cat']].astype(str).sum(axis=1).values

    hazard_prot_sc_df = hazard_prot_sc_.to_dataframe().protection.fillna(0)

    macro_input_vars = ['name', 'gdp_pc_pp', 'pop', 'gini_index', 'avg_prod_k', 'income_share', 'transfers', 'income_group', 'region', 'self_employment', 'ew', 'home_ownership_rate', 'k_pub_share', 'k_household_share']
    macro_inputs_ = macro_res_.to_dataframe()[np.intersect1d(macro_input_vars, macro_sc_.data_vars)]
    macro_inputs_ = pd.merge(macro_inputs_, capital_shares_[['home_ownership_rate', 'k_pub_share']], left_index=True, right_index=True, how='left')
    if 'gini_index' not in macro_inputs_.columns:
        macro_inputs_ = pd.merge(macro_inputs_, gini_index_, left_index=True, right_index=True, how='left')
    macro_inputs_['transfers_share_GDP'] = cat_info_sc_df_[['income_share', 'transfers']].prod(axis=1).groupby(policy_dims + ['iso3']).sum()
    macro_inputs_['policy-iso3'] = macro_inputs_.reset_index()[['policy', 'iso3']].sum(axis=1).values

    fa_avg_unprot = average_over_rp(hazard_ratios_sc_.to_dataframe()['fa']).rename('fa_avg').to_frame()
    fa_avg_unprot['protection'] = False
    fa_avg_unprot = fa_avg_unprot.set_index('protection', append=True)
    fa_avg_prot = average_over_rp(hazard_ratios_sc_.to_dataframe()['fa'], hazard_prot_sc_df).rename('fa_avg').to_frame()
    fa_avg_prot['protection'] = True
    fa_avg_prot = fa_avg_prot.set_index('protection', append=True)
    fa_avg = pd.concat([fa_avg_unprot, fa_avg_prot], axis=0)

    # vulnerability is independent of return period
    hazard_inputs_quintile_ = pd.merge(fa_avg, hazard_ratios_sc_.to_dataframe()[['v', 'v_ew']].dropna().groupby(policy_dims + ['iso3', 'hazard', 'income_cat']).max(),
                                      left_index=True, right_index=True)

    fa_all_hazards = hazard_inputs_quintile_.fa_avg.groupby(policy_dims + ['iso3', 'income_cat', 'protection']).sum()
    v_all_hazards = pd.merge(hazard_inputs_quintile_, cat_info_sc_df_.k, left_index=True, right_index=True, how='left').groupby(policy_dims + ['iso3', 'income_cat', 'protection']).apply(lambda x: np.average(x['v'], weights=x[['fa_avg', 'k']].prod(axis=1))).rename('v')
    v_ew_all_hazards = pd.merge(hazard_inputs_quintile_, cat_info_sc_df_.k, left_index=True, right_index=True, how='left').groupby(policy_dims + ['iso3', 'income_cat', 'protection']).apply(lambda x: np.average(x['v_ew'], weights=x[['fa_avg', 'k']].prod(axis=1))).rename('v_ew')
    fa_v_avg_all_hazards = pd.concat([fa_all_hazards, v_all_hazards, v_ew_all_hazards], axis=1)
    fa_v_avg_all_hazards['hazard'] = 'all hazards'
    fa_v_avg_all_hazards = fa_v_avg_all_hazards.set_index('hazard', append=True).reorder_levels(policy_dims + ['iso3', 'hazard', 'income_cat', 'protection'])
    hazard_inputs_quintile_ = pd.concat([hazard_inputs_quintile_, fa_v_avg_all_hazards], axis=0)

    fa_total_population = hazard_inputs_quintile_.fa_avg.groupby(policy_dims + ['iso3', 'hazard', 'protection']).mean()
    v_total_population = pd.merge(hazard_inputs_quintile_, cat_info_sc_df_.k, left_index=True, right_index=True, how='left').groupby(policy_dims + ['iso3', 'hazard', 'protection']).apply(lambda x: np.average(x.v, weights=x[['fa_avg', 'k']].prod(axis=1))).rename('v')
    v_ew_total_population = pd.merge(hazard_inputs_quintile_, cat_info_sc_df_.k, left_index=True, right_index=True, how='left').groupby(policy_dims + ['iso3', 'hazard', 'protection']).apply(lambda x: np.average(x.v_ew, weights=x[['fa_avg', 'k']].prod(axis=1))).rename('v_ew')
    fa_v_avg_total_population = pd.concat([fa_total_population, v_total_population, v_ew_total_population], axis=1)
    fa_v_avg_total_population['income_cat'] = 'total'
    fa_v_avg_total_population = fa_v_avg_total_population.set_index('income_cat', append=True).reorder_levels(policy_dims + ['iso3', 'hazard', 'income_cat', 'protection'])
    hazard_inputs_quintile_ = pd.concat([hazard_inputs_quintile_, fa_v_avg_total_population], axis=0)

    hazard_inputs_quintile_ = hazard_inputs_quintile_.dropna()
    hazard_inputs_quintile_ = hazard_inputs_quintile_.rename(index={.2: 'q1', .4: 'q2', .6: 'q3', .8: 'q4', 1: 'q5'}, level='income_cat').sort_index()
    hazard_inputs_quintile_['policy-iso3-hazrd-income_cat-protection'] = hazard_inputs_quintile_.reset_index()[['policy', 'iso3', 'hazard', 'income_cat', 'protection']].astype(str).sum(axis=1).values

    model_results_ = cat_info_res_df[['dk', 'dc', 'dw', 'n']].dropna(how='all')
    model_results_ = model_results_.rename(index={.2: 'q1', .4: 'q2', .6: 'q3', .8: 'q4', 1: 'q5'}, level='income_cat')
    model_results_[['dk', 'dc', 'dw']] = model_results_[['dk', 'dc', 'dw']].mul(model_results_.n, axis=0)
    model_results_ = model_results_.groupby(policy_dims + ['iso3', 'hazard', 'rp', 'income_cat']).sum()
    model_results_[['dk', 'dc', 'dw']] = model_results_[['dk', 'dc', 'dw']].div(model_results_.n, axis=0)
    model_results_ = pd.merge(model_results_, hazard_prot_sc_df, left_index=True, right_index=True, how='left')
    model_results_['protection'] = model_results_.protection.values >= model_results_.reset_index().rp.values
    model_results_ = pd.concat([model_results_.drop(columns=['protection', 'n']).mul(~model_results_.protection, axis=0), model_results_.n], axis=1)
    event_results = model_results_[['dk', 'dc', 'dw']].mul(model_results_.n, axis=0).groupby(policy_dims + ['iso3', 'hazard', 'rp']).sum()
    event_results['income_cat'] = 'total'
    event_results['n'] = 1
    event_results = event_results.set_index('income_cat', append=True)
    model_results_ = pd.concat([model_results_, event_results], axis=0).sort_index()
    annual_average_results = pd.concat([average_over_rp(model_results_.drop(columns='n'), hazard_prot_sc_df), model_results_.n.groupby(policy_dims + ['iso3', 'hazard', 'income_cat']).mean()], axis=1)
    annual_average_results['rp'] = 'annual average'
    annual_average_results = annual_average_results.reset_index().set_index(policy_dims + ['iso3', 'hazard', 'rp', 'income_cat'])
    model_results_ = pd.concat([model_results_, annual_average_results], axis=0).sort_index()
    all_hazards_results = model_results_.groupby(policy_dims + ['iso3', 'rp', 'income_cat']).agg({'dk': 'sum', 'dc': 'sum', 'dw': 'sum', 'n': 'mean'})
    all_hazards_results['hazard'] = 'all hazards'
    all_hazards_results = all_hazards_results.reset_index().set_index(policy_dims + ['iso3', 'hazard', 'rp', 'income_cat'])
    model_results_ = pd.concat([model_results_, all_hazards_results], axis=0).sort_index()
    model_results_['dw_currency'] = model_results_['dw'] / (macro_res_.gdp_pc_pp**(-macro_res_.income_elasticity_eta)).to_dataframe(name='currency').currency

    model_results_ = model_results_.rename(columns={'dk': 'dk_pc', 'dc': 'dc_pc', 'dw': 'dw_pc', 'dw_currency': 'dw_pc_currency'})

    model_results_['resilience'] = model_results_.dk_pc / model_results_.dw_pc_currency
    model_results_ = model_results_.dropna()

    model_results_ = model_results_.rename(index={.2: 'q1', .4: 'q2', .6: 'q3', .8: 'q4', 1: 'q5'}, level='income_cat')
    model_results_['policy-iso3-hazard-rp-income_cat'] = model_results_.reset_index()[['policy', 'iso3', 'hazard', 'rp', 'income_cat']].astype(str).sum(axis=1).values
    model_results_['policy-hazard-rp-income_cat'] = model_results_.reset_index()[['policy', 'hazard', 'rp', 'income_cat']].astype(str).sum(axis=1).values

    t_reco_ = calculate_average_recovery_duration(cat_info_res_df, policy_dims + ['iso3', 'hazard', 'income_cat'], hazard_prot_sc_df).to_frame()
    t_reco_income_cat = calculate_average_recovery_duration(cat_info_res_df, policy_dims + ['iso3', 'income_cat'], hazard_prot_sc_df).to_frame()
    t_reco_income_cat['hazard'] = 'all hazards'
    t_reco_income_cat = t_reco_income_cat.reset_index().set_index(policy_dims + ['iso3', 'hazard', 'income_cat'])
    t_reco_ = pd.concat([t_reco_, t_reco_income_cat], axis=0)
    t_reco_hazard = calculate_average_recovery_duration(cat_info_res_df, policy_dims + ['iso3', 'hazard'], hazard_prot_sc_df).to_frame()
    t_reco_hazard['income_cat'] = 'total'
    t_reco_hazard = t_reco_hazard.reset_index().set_index(policy_dims + ['iso3', 'hazard', 'income_cat'])
    t_reco_ = pd.concat([t_reco_, t_reco_hazard], axis=0)
    t_reco_iso3 = calculate_average_recovery_duration(cat_info_res_df, policy_dims + ['iso3'], hazard_prot_sc_df).to_frame()
    t_reco_iso3['hazard'] = 'all hazards'
    t_reco_iso3['income_cat'] = 'total'
    t_reco_iso3 = t_reco_iso3.reset_index().set_index(policy_dims + ['iso3', 'hazard', 'income_cat'])
    t_reco_ = pd.concat([t_reco_, t_reco_iso3], axis=0).sort_index()

    t_reco_ = t_reco_.rename(index={.2: 'q1', .4: 'q2', .6: 'q3', .8: 'q4', 1: 'q5'}, level='income_cat')
    t_reco_['policy-iso3-hazard-income_cat'] = t_reco_.reset_index()[['policy', 'iso3', 'hazard', 'income_cat']].astype(str).sum(axis=1).values

    if excel_outpath is not None:
        with pd.ExcelWriter(excel_outpath, engine='xlsxwriter') as writer:
            workbook = writer.book
            pd.DataFrame().to_excel(writer, sheet_name='Country report', index=False)
            for df_, name in tqdm.tqdm(zip([cat_info_sc_df_, macro_inputs_, hazard_inputs_quintile_, model_results_, t_reco_, data_coverage_.reset_index(), policy_name_lookup_], ['cat_info_inputs', 'macro_inputs', 'hazard_inputs', 'model_results', 't_reco', 'data_coverage', 'policy_name_lookup'])):
                if isinstance(df_, pd.DataFrame):
                    df_ = df_.reset_index()
                else:
                    df_ = df_.to_frame().reset_index()
                df_.to_excel(writer, sheet_name=name, index=False, header=True)
                worksheet = writer.sheets[name]
                worksheet.add_table(0, 0, df_.shape[0], df_.shape[1] - 1, {'name': name, 'columns': [{'header': col} for col in df_.columns], 'style': None})

    return cat_info_sc_df_, macro_inputs_, hazard_inputs_quintile_, model_results_, t_reco_, policy_name_lookup_


def load_model_data(preprocessed_inputs_dir_, raw_data_dir_):
    capital_shares_ = pd.read_csv(os.path.join(preprocessed_inputs_dir_, 'capital_shares.csv'), index_col=0)
    data_coverage_ = pd.read_csv(os.path.join(preprocessed_inputs_dir_, 'data_coverage.csv'), index_col=0)
    gini_index_ = pd.read_csv(os.path.join(raw_data_dir_, 'WB_socio_economic_data/gini_index.csv'), index_col=0) / 100
    return capital_shares_, data_coverage_, gini_index_


def generate_pdf_report(ppp_reference_year, tex_template_path, outpath, countries=None):
    bl_macro_inputs = macro_inputs.loc['baseline/0/+0']
    bl_model_results = model_results.loc['baseline/0/+0']

    if countries is None:
        countries = macro_inputs.index.unique('iso3').values
    for country in tqdm.tqdm(countries, desc="Generating PDF reports"):
        with open(tex_template_path, "r", encoding="utf-8") as f:
            tex_template = f.read()

        tex_template = tex_template.replace("_input/", os.path.join(outpath, f"_input/"))
        tex_template = tex_template.replace("+++iso3+++", country)
        tex_template = tex_template.replace("+++ctry+++", bl_macro_inputs.loc[country, 'name'])
        tex_template = tex_template.replace("+++gdp-per-capita+++", f"\$PPP {bl_macro_inputs.loc[country, 'gdp_pc_pp']:,.2f}")
        tex_template = tex_template.replace("+++country-income-group+++", f"{bl_macro_inputs.loc[country, 'income_group']}")
        tex_template = tex_template.replace("+++world-region+++", f"{WORLD_REGION_NAMES[bl_macro_inputs.loc[country, 'region']]}")
        tex_template = tex_template.replace("+++population+++", f"{int(bl_macro_inputs.loc[country, 'pop']):,.0f}")
        tex_template = tex_template.replace("+++currency-year+++", f"{ppp_reference_year}")
        tex_template = tex_template.replace("+++resilience+++", f"{bl_model_results.loc[(country, 'all hazards', 'annual average', 'total'), 'resilience'] * 100:,.1f}")
        tex_template = tex_template.replace("+++well-being-loss+++", f"{1 / bl_model_results.loc[(country, 'all hazards', 'annual average', 'total'), 'resilience']:,.2f} (= \$1/{bl_model_results.loc[(country, 'all hazards', 'annual average', 'total'), 'resilience']:,.3f})")


        os.makedirs(os.path.join(outpath, f"_input/{country}"), exist_ok=True)

        for f in ['gfdrr_logo.png', 'world_bank_logo.png', 'lib.bib']:
            if not os.path.exists(os.path.join(outpath, f"_input/{country}", f)):
                shutil.copy(os.path.join(Path(tex_template_path).parent, '_input', f), os.path.join(outpath, "_input", f))

        tex_outpath = os.path.join(outpath, f"{country}.tex")
        with open(tex_outpath, "w", encoding="utf-8") as f:
            f.write(tex_template)

        plot_fig_1(macro_res, ['risk_to_assets', 'risk_to_consumption', 'risk_to_wellbeing', 't_reco_95', 'resilience'],
                   country=country, outpath=os.path.join(outpath, f"_input/{country}/{country}_fig_1.pdf"))
        plot_fig_2(macro_inputs, model_results, t_reco, country=country, outpath=os.path.join(outpath, f"_input/{country}/{country}_fig_2.pdf"))
        plot_fig_3(macro_inputs, model_results, country=country, outpath=os.path.join(outpath, f"_input/{country}/{country}_fig_3.pdf"))
        plot_fig_4(macro_res.drop_sel(dict(policy='scale_liquidity/100/-100')), country=country,
                   metrics=['risk_to_assets', 'risk_to_consumption', 'risk_to_wellbeing', 'resilience', 't_reco_95'],
                   scaling_selection=None, outpath=os.path.join(outpath, f"_input/{country}/{country}_fig_4.pdf"))
        plot_fig_5(macro_inputs, country_=country, outpath=os.path.join(outpath, f"_input/{country}/{country}_fig_5.pdf"))
        plot_fig_6(cat_info_inputs, macro_inputs, country_=country, outpath=os.path.join(outpath, f"_input/{country}/{country}_fig_6.pdf"))
        plot_fig_7(hazard_inputs_quintile, country=country, outpath=os.path.join(outpath, f"_input/{country}/{country}_fig_7.pdf"))
        plt.close('all')

        execute = f"cd \"{outpath}\" && latexmk -pdf \"{tex_outpath}\""
        execute_res = os.system(execute)

    #cleanup
    filetypes = ["*.log", "*.aux", "*.out", "*.bcf", "*.blg", "*.bbl", "*.fdb_latexmk", "*.fls", "*.run.xml"]

    for pattern in filetypes:
        for file_path in glob.glob(os.path.join(outpath, pattern)):
            os.remove(file_path)
            print(f"Deleted: {file_path}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate country report')
    parser.add_argument('simulation_outputs_dir', type=str, help='Directory containing simulation outputs')
    parser.add_argument('--tex_template_path', type=str, help='Path to LaTeX template file.')
    parser.add_argument('--ppp_year', type=int, default=2021, help='The PPP reference year.')
    parser.add_argument('--report_outpath', type=str, help='Directory for country report generation.')
    parser.add_argument('--concat_policy_params', action='store_true', help='Whether to concatenate policy parameters to the policy string or keep them as separate variables')
    parser.add_argument('--store_preprocessed', action='store_true', help='Whether to save the preprocessed simulation output to disk.')
    args = parser.parse_args()

    simulation_outputs_dir = args.simulation_outputs_dir
    preprocessed_inputs_dir = os.path.join(simulation_outputs_dir, "_preprocessed_data")
    raw_data_dir = os.path.join(Path(os.path.abspath(__file__)).parent.parent, "scenario/data/raw")
    store_preprocessed = args.store_preprocessed
    concat_policy_params = args.concat_policy_params
    report_outpath = args.report_outpath

    # Preprocess simulation data
    cat_info_res, event_res, macro_res, poverty_res, hazard_prot_sc, cat_info_sc, macro_sc, hazard_ratios_sc = preprocess_simulation_data(simulation_outputs_dir, store_preprocessed, concat_policy_parameters=concat_policy_params)

    # Load preprocessed inputs
    capital_shares, data_coverage, gini_index = load_model_data(preprocessed_inputs_dir, raw_data_dir)

    # Create country report excel file
    cat_info_inputs, macro_inputs, hazard_inputs_quintile, model_results, t_reco, policy_name_lookup = prepare_data(
        cat_info_res_=cat_info_res,
        macro_res_=macro_res,
        hazard_prot_sc_=hazard_prot_sc,
        cat_info_sc_=cat_info_sc,
        macro_sc_=macro_sc,
        hazard_ratios_sc_=hazard_ratios_sc,
        capital_shares_=capital_shares,
        gini_index_=gini_index,
        data_coverage_=data_coverage,
        excel_outpath=None,
    )

    # Generate PDF reports
    if report_outpath is not None:
        generate_pdf_report(
            ppp_reference_year=args.ppp_year,
            tex_template_path=args.tex_template_path,
            outpath=report_outpath
        )