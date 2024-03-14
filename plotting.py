import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.metrics import r2_score
from numpy.polynomial.polynomial import polyfit, polyval
import cartopy.crs as ccrs
from lib import get_country_name_dicts, df_to_iso3


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
    fig, axs = plt.subplots(len(variables), 1, figsize=(12, 6 * len(variables)),
                            subplot_kw=dict(projection=proj))  # , gridspec_kw={'width_ratios': [20, .5]})
    if isinstance(axs, plt.Axes):
        axs = [axs]

    for ax, variable in zip(axs, variables):
        world.boundary.plot(ax=ax, fc='lightgrey', lw=.5, zorder=0, ec='k')
        ax.set_extent([-160, 180, -60, 85])

        if variable in ['resilience', 'risk', 'risk_to_assets']:
            data_[variable] = data_[variable] * 100

        if bins_list[variable] is not None:
            data_.plot(column=variable, ax=ax, legend=show_legend, zorder=5, cmap=cmap[variable], scheme="User_Defined",
                          classification_kwds={'bins': bins_list[variable]})
        else:
            data_.plot(column=variable, ax=ax, legend=show_legend, zorder=5, cmap=cmap[variable])

        format_axis(ax, title=variable, name_mapping=name_dict)

        ax.axis('off')
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=300, bbox_inches='tight')
    if show:
        plt.show(block=False)
    else:
        plt.close()


def plot_scatter(data, x_vars, y_vars, exclude_countries=None, reg_degrees=None, name_dict=None, outfile=None,
                 xlim=None, ylim=None, plot_unit_line=False):
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
        ax.scatter(data[x_var], data[y_var])

        if reg_degree:
            if isinstance(reg_degree, str):
                reg_degree = reg_degrees[reg_degree]
            if isinstance(reg_degree, int) or reg_degree is None:
                reg_degree = [reg_degree]

            for d_idx, (degree, color) in enumerate(zip(reg_degree, ['red', 'blue', 'green', 'orange', 'purple'])):
                # Fit a polynomial to the data
                p = polyfit(data[x_var], data[y_var], degree)
                x_line = np.linspace(min(data[x_var]), max(data[x_var]), 100)
                y_line = polyval(x_line, p)
                ax.plot(x_line, y_line, color=color)

                # Calculate the R-squared value
                y_pred = polyval(data[x_var], p)
                r_squared = r2_score(data[y_var], y_pred)
                ax.text(0.1, 0.9 - .05 * d_idx, r'$R^2 = $ {:.3f}'.format(r_squared), transform=ax.transAxes,
                        color=color)

        format_axis(ax, x_name=x_var, y_name=y_var, name_mapping=name_dict, xlim=xlim.get(x_var), ylim=ylim.get(y_var))

        if plot_unit_line:
            ax.axline([0, 0], slope=1, ls='--', c='k', lw=.5)

    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.show(block=False)


def plot_hbar(data_path, variables, comparison_data_path=None, norm=None, how='abs', head=15, outfile=None,
              name_dict=None, unit=None, precision=None):
    if isinstance(variables, str):
        variables = [variables]
    if precision is None:
        precision = 2
    data = pd.read_csv(data_path)
    if comparison_data_path is not None:
        comparison_data = pd.read_csv(comparison_data_path)
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
        xlabel = f'{xlabel} (millions USD)'
    elif unit == 'billions':
        data[variables] = data[variables] / 1e9
        xlabel = f'{xlabel} (billions USD)'

    fig, ax = plt.subplots(1, 1, figsize=(4, 6))
    bars = data[variables[::-1]].rename(columns=name_dict).plot(kind='barh', ax=ax, legend=False, width=.8)

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


if __name__ == '__main__':
    results_path = './output/scenarios/baseline/results_tax_unif_poor.csv'
    pov_reduction_results_path = "./output/scenarios/reduce_poor_exposure/results_tax_unif_poor.csv"
    name_dict = {
        'resilience': 'socio-economic resilience (%)',
        'risk': 'risk to well-being (% of GDP)',
        'risk_to_assets': 'risk to assets (% of GDP)',
        'gdp_pc_pp': 'GDP per capita (PPP USD)',
        'dk_tot': 'Total asset losses',
        'dWtot_currency': 'Welfare losses (currency)',
    }
    any_to_wb, iso3_to_wb, iso2_iso3 = get_country_name_dicts("./")
    plot_map(
        data=results_path,
        variables=['resilience', 'risk', 'risk_to_assets'],
        # bins_list={'resilience': [25, 51, 59, 65, 72, 81], 'risk': [.3, .5, .8, 1.5, 6.55],
        #            'risk_to_assets': [.2, .3, .5, .9, 4.5]},
        name_dict=name_dict,
        cmap={'resilience': 'YlOrBr_r', 'risk': 'Blues', 'risk_to_assets': 'Greens'},
        outfile="./figures/resilience-wellbeing_risk-asset_risk_map.pdf",
    )
    plot_scatter(
        data=results_path,
        x_vars=['gdp_pc_pp', 'gdp_pc_pp', 'gdp_pc_pp'],
        y_vars=['resilience', 'risk', 'risk_to_assets'],
        reg_degrees={'resilience': [1, 2], 'risk': [1, 2], 'risk_to_assets': [1, 2]},
        name_dict=name_dict,
        ylim={'resilience': (0, .65)},
        outfile="./figures/GDP_vs-resilience-wellbeing_risk-asset_risk_scatter.pdf",
    )
    plot_hbar(
        data_path=results_path,
        comparison_data_path=pov_reduction_results_path,
        variables=["dWtot_currency", "dk_tot"],
        how='abs',
        unit='millions',
        name_dict=name_dict,
        precision=0,
        outfile="./figures/dW-dk_comparison_abs.pdf",
    )
    plot_hbar(
        data_path=results_path,
        comparison_data_path=pov_reduction_results_path,
        variables=["dWtot_currency", "dk_tot"],
        how='rel',
        name_dict=name_dict,
        outfile="./figures/dW-dk_comparison_rel.pdf",
    )
    plot_hbar(
        data_path=results_path,
        comparison_data_path=None,
        variables=["dWtot_currency", "dk_tot"],
        unit='billions',
        precision=0,
        name_dict=name_dict,
        outfile="./figures/dW-dk_abs.pdf",
    )
    plot_hbar(
        data_path=results_path,
        comparison_data_path=None,
        variables=["dWtot_currency", "dk_tot"],
        norm='GDP',
        name_dict=name_dict,
        outfile="./figures/dW-dk_GDP_rel.pdf",
    )
    old_results_path = "./__legacy_structure/output/original_output/results_tax_unif_poor_.csv"
    old_new_data = pd.merge(pd.read_csv(results_path).set_index('iso3'),
                            df_to_iso3(pd.read_csv(old_results_path), 'country').set_index('iso3'),
                            left_index=True, right_index=True, suffixes=('_new', '_old'))
    plot_scatter(
        data=old_new_data,
        x_vars=['resilience_old', 'risk_old', 'risk_to_assets_old'],
        y_vars=['resilience_new', 'risk_new', 'risk_to_assets_new'],
        outfile="./figures/old_new_comparison.pdf",
        plot_unit_line=True,
    )