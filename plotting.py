import argparse
import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from matplotlib.gridspec import GridSpec
from sklearn.metrics import r2_score
from numpy.polynomial.polynomial import polyfit, polyval
import cartopy.crs as ccrs
from lib import get_country_name_dicts, df_to_iso3
from lib_compute_resilience_and_risk import agg_to_event_level
from lib_prepare_scenario import average_over_rp
from pandas_helper import load_input_data
from recovery_optimizer import baseline_consumption_c_h, delta_c_h_of_t, delta_k_h_eff_of_t
from wb_api_wrapper import get_wb_series
import seaborn as sns


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


def plot_map(data, variables=None, exclude_countries=None, bins_list=None, cmap='viridis', name_dict=None, outfile=None,
             show=False, show_legend=True):
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres')).set_crs(4326).to_crs('World_Robinson')
    world = world[~world.continent.isin(['Antarctica', 'seven seas (open ocean)'])]
    if isinstance(data, str):
        data = pd.read_csv(data)
    elif not isinstance(data, (pd.DataFrame, pd.Series, gpd.GeoDataFrame)):
        raise ValueError('data should be a path to a csv file, a pandas DataFrame, a pandas Series, or a GeoDataFrame.')
    if 'iso3' not in list(data.columns if isinstance(data, pd.DataFrame) else []) + list(data.index.names):
        if 'country' in list(data.columns if isinstance(data, pd.DataFrame) else []) + list(data.index.names):
            data = df_to_iso3(data.reset_index(), 'country').set_index('iso3').copy()
        else:
            raise ValueError('Neither "iso3" nor "country" were found in the data.')
    data_ = gpd.GeoDataFrame(pd.merge(data, world.rename(columns={'iso_a3': 'iso3'}), on='iso3', how='inner'))
    if len(data_) != len(data):
        print(f'{len(data) - len(data_)} countries were not found in the world map. They will be excluded from the plot.')
    data_.replace([np.inf, -np.inf], np.nan, inplace=True)

    if isinstance(variables, str):
        variables = [variables]
    elif variables is None:
        if isinstance(data_, pd.Series):
            data_ = data_.to_frame()
        variables = list(set(data_.columns) - set(world.columns) - {'iso3', 'country'})
    if bins_list is None:
        bins_list = {v: None for v in variables}
    if isinstance(cmap, str):
        cmap = {v: cmap for v in variables}
    elif not isinstance(cmap, dict):
        raise ValueError('cmap should be a string or a dictionary.')

    # data_.dropna(subset=variables, inplace=True)

    if exclude_countries:
        if isinstance(exclude_countries, str):
            exclude_countries = [exclude_countries]
        data_ = data_[~data_.iso3.isin(exclude_countries)]

    proj = ccrs.Robinson(central_longitude=0, globe=None)
    # fig, axs = plt.subplots(nrows=len(variables), ncols=2, figsize=(8, 4 * len(variables)),
    #                         subplot_kw=dict(projection=proj), gridspec_kw={'width_ratios': [20, 2]})

    nrows = len(variables)
    ncols = 2
    fig = plt.figure(figsize=(8, 4 * nrows))
    gs = GridSpec(nrows, ncols, figure=fig, width_ratios=[20, .5])
    axs = [[fig.add_subplot(gs[i, j], projection=(proj if j % 2 == 0 else None)) for j in range(ncols)] for i in range(nrows)]

    for (ax, cax), variable in zip(axs, variables):
        world.boundary.plot(ax=ax, fc='lightgrey', lw=.5, zorder=0, ec='k')
        ax.set_extent([-160, 180, -60, 85])

        if bins_list[variable] is not None:
            data_.plot(column=variable, ax=ax, legend=show_legend, zorder=5, cmap=cmap[variable], scheme="User_Defined",
                          classification_kwds={'bins': bins_list[variable]}, legend_kwds={'loc': 'lower left', 'bbox_to_anchor': (0, 0)})
            cax.set_visible(False)
        else:
            data_.plot(column=variable, ax=ax, legend=show_legend, cmap=cmap[variable], cax=cax)

        format_axis(ax, title=variable, name_mapping=name_dict)

        ax.axis('off')
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=300, bbox_inches='tight', transparent=True)
    if show:
        plt.show(block=False)
    else:
        plt.close()
    return axs


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
        xlabel = f'{xlabel} (m USD)'
    elif unit == 'billions':
        data[variables] = data[variables] / 1e9
        xlabel = f'{xlabel} (bn USD)'

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


def load_data_per_income_cat(path, scenario='Existing_climate/baseline'):
    cat_info = pd.read_csv(os.path.join(path, 'iah_tax_unif_poor.csv'), index_col=[0, 1, 2, 3, 4, 5])
    protection = pd.read_csv(f"./intermediate/scenarios/{scenario}/scenario__hazard_protection.csv", index_col=[0, 1])
    income_cat_data = cat_info[['c']].droplevel(['hazard', 'rp', 'affected_cat', 'helped_cat']).drop_duplicates()
    for variable in ['dk', 'dw']:
        var_data = agg_to_event_level(cat_info, variable, ['iso3', 'hazard', 'rp', 'income_cat'])
        var_data = average_over_rp(var_data, 'default_rp', protection).groupby(['iso3', 'income_cat']).sum()
        income_cat_data[variable] = var_data
        income_cat_data[f'{variable}_rel'] = var_data / var_data.groupby('iso3').sum() * 100
    income_cat_data['dw_pc_currency'] = income_cat_data['dw'] / income_cat_data.c.groupby('iso3').mean() ** (-1.5)
    income_cat_data['resilience'] = income_cat_data['dk'] / income_cat_data['dw_pc_currency'] * 100
    income_cat_data['dw_over_dk'] = income_cat_data['dw_pc_currency'] / income_cat_data['dk']
    return income_cat_data


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


def load_income_groups():
    income_groups_ = load_input_data("./", 'WB_country_classification/country_classification.xlsx', header=0)[["Code", "Region", "Income group"]]
    income_groups_ = income_groups_.dropna().rename({'Code': 'iso3'}, axis=1)
    income_groups_ = income_groups_.set_index('iso3').squeeze()
    income_groups_.loc['VEN'] = ['Latin America & Caribbean', 'Upper middle income']
    return income_groups_


def make_income_cat_boxplots(results_path, outpath=None, focus_countries=None):
    income_cat_data = load_data_per_income_cat(results_path, scenario=f"{climate_scenario}/baseline_EW-2018")
    income_cat_data = income_cat_data.join(income_groups)
    income_cat_data.replace({'Sub-Saharan Africa': 'SSA', 'Latin America & Caribbean': 'LAC',
                             'Middle East & North Africa': 'MENA', 'Europe & Central Asia': 'ECA', 'South Asia': 'SA',
                             'North America': 'NAR', 'East Asia & Pacific': 'EAP'}, inplace=True)
    income_cat_data.replace({'Low income': 'Low', 'Lower middle income': 'Lower middle',
                             'Upper middle income': 'Upper middle', 'High income': 'High'}, inplace=True)
    income_cat_data['income_group_order'] = income_cat_data['Income group'].replace({'Low': 0, 'Lower middle': 1,
                                                                                     'Upper middle': 2, 'High': 3})
    income_cat_data.index.names = ['iso3', 'income quintile']
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    sns.boxplot(income_cat_data[['dw_rel', 'dk_rel']].stack().reset_index().rename(
        {'level_2': 'loss', 0: 'share (%)', }, axis=1).replace(
        {'dw_rel': 'welfare', 'dk_rel': 'assets'}), x='income quintile', y='share (%)', hue='loss', fliersize=0,
        linewidth=.5, palette=[plt.get_cmap('Blues')(.5), plt.get_cmap('Greens')(.5)]
    )
    plt.tight_layout()
    if outpath is not None:
        plt.savefig(f"{outpath}/dk_dw_income_cat_boxplots_WORLD.pdf", dpi=300, bbox_inches='tight')

    for groupby in ['Region', 'Income group']:
        fig, axs = plt.subplots(ncols=2, figsize=(9, 4.5), sharex=True, sharey=True)
        if groupby == 'Income group':
            order = ['Low', 'Lower middle', 'Upper middle', 'High']
        else:
            order = None
        sns.boxplot(income_cat_data.reset_index(), x=groupby, y='dk_rel', hue='income quintile', ax=axs[0], fliersize=0,
                    linewidth=.5, palette='Greens', order=order, legend=False)
        sns.boxplot(income_cat_data.reset_index(), x=groupby, y='dw_rel', hue='income quintile', ax=axs[1], fliersize=0,
                    linewidth=.5, palette='Blues', order=order, legend='brief')
        axs[0].set_ylabel('share (%)')
        axs[0].set_title('share of asset losses')
        axs[1].set_title('share of welfare losses')
        plt.tight_layout()
        if outpath is not None:
            plt.savefig(f"{outpath}/dk_dw_income_cat_boxplots_{groupby}.pdf", dpi=300, bbox_inches='tight')

    if focus_countries is not None:
        if isinstance(focus_countries, str):
            focus_countries = [focus_countries]

        nrows = len(focus_countries)
        fig = plt.figure(figsize=(7, 1.75 * nrows))
        gs = GridSpec(nrows, 2, figure=fig)
        axs_l = []
        axs_r = []
        for i in range(nrows):
            axs_l.append(fig.add_subplot(gs[i, 0], sharex=axs_l[0] if i > 0 else None, sharey=axs_l[0] if i > 0 else None))
            axs_r.append(fig.add_subplot(gs[i, 1], sharex=axs_r[0] if i > 0 else None, sharey=None))
        for ax in axs_r:
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
        for ax_l, ax_r, country in zip(axs_l, axs_r, focus_countries):
            income_cat_data.loc[country, 'dw_over_dk'].plot.bar(ax=ax_r, width=.75, legend=False)
            (income_cat_data.loc[country, 'c'] / income_cat_data.loc[country, 'c'].sum() * 100).rename('income distribution').plot.bar(ax=ax_l, width=.75, legend=False)
            ax_r.axhline(income_cat_data.loc[country, 'dw_pc_currency'].sum() / income_cat_data.loc[country, 'dk'].sum(), color='r', lw=1.25, ls='--')
            # (income_cat_data.loc[country, ['dw', 'dk']] / income_cat_data.loc[country, ['dw', 'dk']].sum() * 100).rename(columns={'dk': 'asset loss', 'dw': 'welfare loss'}).plot.bar(ax=ax_r, color=[plt.get_cmap('Blues')(.5), plt.get_cmap('Greens')(.5)], width=.75, legend=True if ax_r == axs_r[0] else False)
            ax_l.text(-.25, .5, country, transform=ax_l.transAxes, ha='right')

            print("\n#### Country: ", country)
            print((income_cat_data.loc[country, 'c'] / income_cat_data.loc[country, 'c'].sum() * 100).rename('income distribution'))
            print(income_cat_data.loc[country, 'dw_over_dk'])
            print("Average welfare loss per $1 asset loss: ", income_cat_data.loc[country, 'dw_pc_currency'].sum() / income_cat_data.loc[country, 'dk'].sum())
        axs_l[0].set_title('Income distribution')
        axs_r[0].set_title('Welfare loss per $1 asset loss')
        axs_l[-1].set_xlabel(None)
        axs_r[-1].set_xlabel(None)
        axs_l[1].set_ylabel('share (%)')
        axs_r[1].set_ylabel('welfare loss (\$)')
        y_min = float('inf')
        y_max = float('-inf')
        for ax in axs_r:
            curr_y_min, curr_y_max = ax.get_ylim()
            y_min = min(y_min, curr_y_min)
            y_max = max(y_max, curr_y_max)
        for ax in axs_r:
            ax.set_ylim(y_min, y_max)
        axs_l[-1].set_xlabel('income quintile')
        axs_r[-1].set_xlabel('income quintile')
        plt.tight_layout()
        if outpath is not None:
            plt.savefig(f"{outpath}/income_dw_dk_dist_{'-'.join(focus_countries)}.pdf", dpi=300, bbox_inches='tight')


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
    ax.set_ylabel('Avoided losses (bn USD)')
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
    merged['income_group_order'] = merged['Income group'].replace({'Low': 0, 'Lower middle': 1,
                                                                     'Upper middle': 2, 'High': 3})
    merged = merged.sort_values('income_group_order').reset_index(drop=True)
    fig1, ax1 = plt.subplots(figsize=(4.5, 4.3))
    sns.boxplot(merged, x='Income group', y='resilience', palette='tab10')
    ax1.set_ylabel('resilience (%)')
    plt.tight_layout()

    # plot resilience vs GDP per capita, coloring by income group
    fig2, ax2 = plt.subplots(figsize=(4.5, 4.3))
    sns.scatterplot(data=merged, x='gdp_pc_pp', y='resilience', hue='Income group', palette='tab10', alpha=.5)
    ax2.set_xlabel('GDP per capita (PPP USD)')
    ax2.set_ylabel('resilience (%)')
    plt.tight_layout()
    fig3, ax3 = plt.subplots(figsize=(4.5, 4.3))
    sns.scatterplot(data=merged, x='gdp_pc_pp', y='resilience', alpha=.5)
    ax3.set_xlabel('GDP per capita (PPP USD)')
    ax3.set_ylabel('resilience (%)')
    plt.tight_layout()

    # plot resilience vs gini index
    fig4, ax4 = plt.subplots(figsize=(4.5, 4.3))
    sns.scatterplot(data=merged, x='gini_index', y='resilience', alpha=.5)#, hue='Income group', palette='tab10')
    ax4.set_xlabel('Gini index (%)')
    ax4.set_ylabel('resilience (%)')
    plt.tight_layout()
    if outpath:
        fig4.savefig(os.path.join(outpath, "resilience_vs_gini_index_scatter_not_annotated.pdf"), dpi=300)

    # plot Risk to assets vs risk to assets
    fig5, ax5 = plt.subplots(figsize=(4.5, 4.3))
    sns.scatterplot(data=merged, x='gdp_pc_pp', y='risk_to_assets', alpha=.5)  # , hue='Income group', palette='tab10')
    ax5.set_xlabel('GDP per capita (PPP USD)')
    ax5.set_ylabel('risk to assets (% GDP)')
    plt.tight_layout()

    # plot Risk vs risk to assets
    fig6, ax6 = plt.subplots(figsize=(4.5, 4.3))
    sns.scatterplot(data=merged, x='gdp_pc_pp', y='risk', alpha=.5)  # , hue='Income group', palette='tab10')
    ax6.set_xlabel('GDP per capita (PPP USD)')
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
    ax.set_title('Socioeconomic resilience (%)')
    plt.tight_layout()
    if outpath_:
        plt.savefig(os.path.join(outpath_, f"{'pds' if is_pds else 'liquidity'}_comparison_scatter.pdf"), dpi=300, bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(4.5, 4.3))
    sns.boxplot(data=results_merged.rename(columns=rename), x='Income group', y='Avoided wellbeing losses (%)', palette='tab10',
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
    asset_losses = average_over_rp(agg_to_event_level(asset_losses, 'dk', ['iso3', 'hazard', 'rp', 'income_cat']), 'default_rp', None).groupby(['iso3', 'income_cat']).sum()

    merged = pd.merge(asset_losses.rename('Asset loss'), liquidity.rename('liquid savings'), left_index=True, right_index=True, how='inner')
    merged = pd.merge(merged, income_groups, left_on='iso3', right_index=True, how='left')
    merged['Asset losses as share of liquid savings (%)'] = merged['Asset loss'] / merged['liquid savings'] * 100
    merged.index.names = ['iso3', 'Income quintile']
    merged.replace({'High income': 'High', 'Upper middle income': 'Upper middle', 'Lower middle income': 'Lower middle', 'Low income': 'Low'}, inplace=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=merged, x='Income group', y='Asset losses as share of liquid savings (%)', palette='tab10', hue='Income quintile',
                fliersize=0, order=['Low', 'Lower middle', 'Upper middle', 'High'])
    ax.legend(loc='upper left', title='Income quintile', bbox_to_anchor=(1, 1))


def plot_recovery(t_max, productivity_pi_, delta_tax_sp_, k_h_eff_, delta_k_h_eff_, lambda_h_, sigma_h_, savings_s_h_,
                  delta_i_h_pds_, delta_c_h_max_, recovery_params_, social_protection_share_gamma_h_, diversified_share_,
                  show_sp_losses=False, consumption_floor_xi_=None, t_hat_=None, t_tilde_=None, delta_tilde_k_h_eff_=None,
                  consumption_offset_=None, title=None, ylims=None):
    """
    Make a plot of the consumption and capital losses over time
    """
    fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(7, 5))

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
        axs[0].fill_between(t_, c_baseline, c_baseline - di_h_sp, color='red', alpha=0.75, label='Transfers loss', lw=0)
    axs[0].fill_between(t_, c_baseline - di_h_sp, c_baseline - di_h, color='red', alpha=0.5,
                       label='Income loss', lw=0)
    axs[0].fill_between(t_, c_baseline - di_h, c_baseline - (di_h + dc_reco), color='red', alpha=0.25,
                        label='Reconstruction loss', lw=0)
    axs[0].fill_between(t_[dc_savings_pds != 0], (c_baseline - (di_h + dc_reco) + dc_savings_pds)[dc_savings_pds != 0],
                       (c_baseline - (di_h + dc_reco))[dc_savings_pds != 0], facecolor='none', lw=0, hatch='XXX',
                        edgecolor='grey', label='Liquid savings and PDS')
    axs[0].plot([-0.03 * (max(t_) - min(t_)), 0], [c_baseline, c_baseline], color='black', label='__none__')
    axs[0].plot([0, 0], [c_baseline, (c_baseline - di_h - dc_reco + dc_savings_pds)[0]], color='black', label='__none__')
    axs[0].plot(t_, c_baseline - di_h - dc_reco + dc_savings_pds, color='black', label='Consumption')

    dk_eff = delta_k_h_eff_of_t(t_, 0, delta_k_h_eff_, lambda_h_, sigma_h_, delta_c_h_max_, productivity_pi_)
    axs[1].fill_between(t_, 0, dk_eff, color='red', alpha=0.5, label='Effective capital loss')
    axs[1].plot([-0.03 * (max(t_) - min(t_)), 0], [0, 0], color='black', label='__none__')
    axs[1].plot([0, 0], [0, dk_eff[0]], color='black', label='__none__')
    axs[1].plot(t_, dk_eff, color='black', label='Effective capital loss')

    axs[1].set_xlabel('Time [y]')
    axs[0].set_ylabel(r'Consumption $c(t)$')
    axs[1].set_ylabel(r'Capital loss $\Delta k(t)$')

    if ylims is not None:
        axs[0].set_ylim(ylims[0])
        axs[1].set_ylim(ylims[1])

    if title is not None:
        axs[0].set_title(title)
    for ax in axs:
        ax.legend(frameon=False, bbox_to_anchor=(1, 1), loc='upper left')
    plt.tight_layout()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script parameters')
    parser.add_argument('--climate_scenario', type=str, default='Existing climate')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()

    climate_scenario = args.climate_scenario.replace(' ', '_')
    outpath = f"./figures/{climate_scenario}"
    os.makedirs(outpath, exist_ok=True)

    datasets = {
        'baseline': pd.read_csv("./output/scenarios/Existing_climate/2024-05-14_15-20_baseline_EW-2018/results_tax_unif_poor.csv"),
        'poor_reduction_results': pd.read_csv("./output/scenarios/Existing_climate/2024-05-15_09-57_reduce_poor_exposure_5_EW-2018/results_tax_unif_poor.csv"),
        'nonpoor_reduction_results': pd.read_csv("./output/scenarios/Existing_climate/2024-05-15_10-08_reduce_nonpoor_exposure_5_EW-2018/results_tax_unif_poor.csv"),
        'no_liquidity_no_pds': pd.read_csv("./output/scenarios/Existing_climate/2024-05-15_11-32_no_liquidity_EW-2018_noPDS/results_tax_no.csv"),
        'with_liquidity_no_pds': pd.read_csv("./output/scenarios/Existing_climate/2024-05-15_11-43_baseline_EW-2018_noPDS/results_tax_no.csv"),
        'no_liquidity_with_pds': pd.read_csv("./output/scenarios/Existing_climate/2024-05-15_11-36_no_liquidity_EW-2018/results_tax_unif_poor.csv"),
    }

    gini_index = get_wb_series('SI.POV.GINI').rename('gini_index').dropna().reset_index()
    gini_index = gini_index.loc[gini_index.groupby('country').year.idxmax()].drop(columns='year')
    gini_index = df_to_iso3(gini_index, 'country').set_index('iso3').drop(columns='country').squeeze()

    for k in datasets.keys():
        datasets[k] = datasets[k][datasets[k].iso3 != 'THA']
        datasets[k][['resilience', 'risk', 'risk_to_assets']] *= 100
        datasets[k] = datasets[k].join(gini_index, on='iso3')

    income_groups = load_income_groups()

    name_dict = {
        'resilience': 'socio-economic resilience (%)',
        'risk': 'risk to well-being (% of GDP)',
        'risk_to_assets': 'risk to assets (% of GDP)',
        'gdp_pc_pp': 'GDP per capita (PPP USD)',
        'dk_tot': 'Asset losses',
        'dWtot_currency': 'Welfare losses',
        'gini_index': 'Gini index (%)',
    }
    any_to_wb, iso3_to_wb, iso2_iso3 = get_country_name_dicts("./")

    if args.plot:
        # plot recovery of ('COL', 'Earthquake', 5000, 'q1', 'a', 'not_helped') for scenarios with and without savings
        for directory, with_savings in zip(["2024-05-15_11-32_no_liquidity_EW-2018_noPDS", "2024-05-15_11-43_baseline_EW-2018_noPDS"], [False, True]):
            cat_info = pd.read_csv(f"./output/scenarios/Existing_climate/{directory}/iah_tax_no.csv", index_col=[0, 1, 2, 3, 4, 5])
            macro = pd.read_csv(f"./output/scenarios/Existing_climate/{directory}/macro_tax_no.csv", index_col=[0, 1, 2])
            data = pd.merge(macro, cat_info, left_index=True, right_index=True)
            data = data.loc[('COL', 'Earthquake', 5000, 'q1', 'a', 'not_helped')]
            data.recovery_params = [(float(d.split(', ')[0]), float(d.split(', ')[1])) for d in data.recovery_params[2:-2].split('), (')]
            plot_recovery(10, data.avg_prod_k, data.tau_tax, data.k, data.dk,
                          data.lambda_h, data.reconstruction_share_sigma_h, data.liquidity, data.help_received, np.nan,
                          data.recovery_params, data.gamma_SP * data.n, data.diversified_share,
                          ylims=[(-2200, 3000), None], title=f"{'with' if with_savings else 'without'} liquid savings and borrowing")
            plt.savefig(f"./figures/{climate_scenario}/COL_Earthquake_5000_q1_a_not-helped_{'with' if with_savings else 'without'}-liquidity.pdf", dpi=300, bbox_inches='tight')

        plot_map(
            data=datasets['baseline'],
            variables=['risk_to_assets', 'risk', 'resilience'],
            bins_list={'resilience': None, 'risk': [.2, .4, 1, 2, 6],
                       'risk_to_assets': [.15, .3, .5, 1, 3]},
            name_dict=name_dict,
            cmap={'resilience': 'YlOrBr_r', 'risk': 'Blues', 'risk_to_assets': 'Greens'},
            outfile=f"{outpath}/resilience-wellbeing_risk-asset_risk_map.pdf",
        )
        plot_map(
            data=datasets['baseline'],
            variables='risk_to_assets',
            bins_list={'risk_to_assets': [.15, .3, .5, 1, 3]},
            name_dict=name_dict,
            cmap='Greens',
            outfile=f"{outpath}/risk_to_assets_map.pdf",
        )
        plot_map(
            data=datasets['baseline'],
            variables='risk',
            bins_list={'risk': [.2, .4, 1, 2, 6]},
            name_dict=name_dict,
            cmap='Blues',
            outfile=f"{outpath}/risk_map.pdf",
        )
        plot_map(
            data=datasets['baseline'],
            variables='resilience',
            bins_list={'resilience': None},  # , 'risk': [.2, .4, 1, 2, 6],
            name_dict=name_dict,
            cmap='YlOrBr_r',
            outfile=f"{outpath}/resilience_map.pdf",
        )

        plot_drivers_gdp_and_gini_index(
            datasets['baseline'],
            outpath=outpath,
            annotate=['HTI', 'LAO', 'HND', 'TJK', 'GRC', 'MMR', 'URK', 'ECU', 'BTN', 'IRL', 'LUX', 'UKR', 'IRN', 'GEO']
        )

        make_income_cat_boxplots(
            results_path="./output/scenarios/Existing_climate/2024-05-14_15-20_baseline_EW-2018",
            outpath=outpath,
            focus_countries=['FIN', 'BOL', 'NAM'],
        )

        plot_hbar(
            data=datasets['baseline'],
            comparison_data=datasets['poor_reduction_results'],
            variables=["dWtot_currency", "dk_tot"],
            how='abs',
            unit='millions',
            name_dict=name_dict,
            precision=0,
            outfile=f"{outpath}/poor_exposure_reduction_dW-dk_comparison_abs.pdf",
        )
        plot_hbar(
            data=datasets['baseline'],
            comparison_data=datasets['poor_reduction_results'],
            variables=["dWtot_currency", "dk_tot"],
            how='rel',
            name_dict=name_dict,
            outfile=f"{outpath}/poor_exposure_reduction_dW-dk_comparison_rel.pdf",
        )
        plot_hbar(
            # data=datasets['baseline'][datasets['baseline'].iso3.isin(['CHN', 'IND', 'IDN', 'USA', 'ITA', 'TUR', 'JPN', 'VNM', 'IRN', 'PAK', 'GRC', 'KOR', 'COL', 'PHL', 'URK'])],
            data=datasets['baseline'],
            comparison_data=datasets['nonpoor_reduction_results'],
            variables=["dWtot_currency", "dk_tot"],
            how='abs',
            unit='millions',
            name_dict=name_dict,
            precision=0,
            outfile=f"{outpath}/nonpoor_exposure_reduction_dW-dk_comparison_abs.pdf",
        )
        plot_hbar(
            # data=datasets['baseline'][datasets['baseline'].iso3.isin(['AZE', 'NGA', 'ALB', 'SVK', 'IDN', 'NPL', 'HND', 'CYP', 'GHA', 'HTI', 'LAO', 'COL', 'AGO', 'PAK', 'IND'])],
            data=datasets['baseline'],
            comparison_data=datasets['nonpoor_reduction_results'],
            variables=["dWtot_currency", "dk_tot"],
            how='rel',
            name_dict=name_dict,
            outfile=f"{outpath}/nonpoor_exposure_reduction_dW-dk_comparison_rel.pdf",
        )
        make_liquidity_comparison_plot(
            path_with="./output/scenarios/Existing_climate/2024-05-15_11-43_baseline_EW-2018_noPDS/",
            path_without="./output/scenarios/Existing_climate/2024-05-15_11-32_no_liquidity_EW-2018_noPDS/",
            is_pds=False,
            outpath_=outpath,
        )
        make_liquidity_comparison_plot(
            path_with="./output/scenarios/Existing_climate/2024-05-15_11-36_no_liquidity_EW-2018/",
            path_without="./output/scenarios/Existing_climate/2024-05-15_11-32_no_liquidity_EW-2018_noPDS/",
            is_pds=True,
            outpath_=outpath,
        )
        plot_ew_impact_over_time(
            ensemble_dir="./output/scenarios_pre_max_aid_fix/Existing_climate",
            outpath_=outpath,
        )

