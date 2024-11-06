import argparse
import itertools
import os

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import BoundaryNorm, LogNorm
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from matplotlib.transforms import blended_transform_factory

import statsmodels.api as sm

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.metrics import r2_score
from numpy.polynomial.polynomial import polyfit, polyval
import cartopy.crs as ccrs
from lib import get_country_name_dicts, df_to_iso3
from lib_compute_resilience_and_risk import agg_to_event_level
from lib_prepare_scenario import average_over_rp
from pandas_helper import load_input_data
from prepare_scenario import calc_reconstruction_share_sigma
from recovery_optimizer import baseline_consumption_c_h, delta_c_h_of_t, delta_k_h_eff_of_t
from wb_api_wrapper import get_wb_series
import seaborn as sns

INCOME_GROUP_COLORS = {
    'L': plt.get_cmap('Set1')(0),
    'LM': plt.get_cmap('Set1')(1),
    'UM': plt.get_cmap('Set1')(2),
    'H': plt.get_cmap('Set1')(3),
}

INCOME_GROUP_MARKERS = ['o', 's', 'D', '^']

# Set the default font size for labels and ticks
plt.rcParams['axes.labelsize'] = 7  # Font size for x and y labels
plt.rcParams['xtick.labelsize'] = 6  # Font size for x tick labels
plt.rcParams['ytick.labelsize'] = 6  # Font size for y tick labels
plt.rcParams['legend.fontsize'] = 6  # Font size for legend
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


def add_regression(ax_, data_, x_var_, y_var, p_val_pos='above left'):
    # Perform OLS regression
    reg_data = data_.sort_values(by=x_var_).copy()
    X = sm.add_constant(reg_data[x_var_])
    y = reg_data[y_var]
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
    elif p_val_pos == 'above right':
        xy = (1, 1.01)
    elif p_val_pos == 'lower left':
        xy = (.01, .01)
    ax_.annotate(f'R2={r2:.2f}{significance}', xy=xy, xycoords='axes fraction',
                 ha='left', va='bottom', fontsize=6)


def plot_fig_1(data_, exclude_countries=None, bins_list=None, cmap='viridis', outfile=None,
               show=False, numbering=True, annotate=None, run_ols=False, log_xaxis=False):
    """
    Plots a map with the given data and variables.

    Parameters:
    data_ (str or DataFrame or Series or GeoDataFrame): The data to plot. Can be a path to a CSV file, a pandas DataFrame,
                                                        a pandas Series, or a GeoDataFrame.
    exclude_countries (list or str, optional): Countries to exclude from the plot. Defaults to None.
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

    name_dict_ = {
        'resilience': 'socio-economic\nresilience [%]',
        'risk': 'wellbeing losses\n[% of GDP]',
        'risk_to_assets': 'asset losses\n[% of GDP]',
        'gdp_pc_pp': 'GDPpc [1000 PPP USD]',
        'log_gdp_pc_pp': 'log(GDPpc)',
        'dk_tot': 'Asset losses\n[PPP USD]',
        'dWtot_currency': 'Welfare losses\n[PPP USD]',
        'gini_index': 'Gini index [%]',
    }

    # Load and prepare the world map
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres')).set_crs(4326).to_crs('World_Robinson')
    world = world[~world.continent.isin(['Antarctica', 'seven seas (open ocean)'])]

    # Load data
    if isinstance(data_, str):
        data_ = pd.read_csv(data_)
    elif not isinstance(data_, (pd.DataFrame, pd.Series, gpd.GeoDataFrame)):
        raise ValueError('data should be a path to a csv file, a pandas DataFrame, a pandas Series, or a GeoDataFrame.')

    # Ensure data has 'iso3' column
    if 'iso3' not in list(data_.columns if isinstance(data_, pd.DataFrame) else []) + list(data_.index.names):
        if 'country' in list(data_.columns if isinstance(data_, pd.DataFrame) else []) + list(data_.index.names):
            data_ = df_to_iso3(data_.reset_index(), 'country').set_index('iso3').copy()
        else:
            raise ValueError('Neither "iso3" nor "country" were found in the data.')

    # Merge data with world map
    data_ = gpd.GeoDataFrame(pd.merge(data_, world.rename(columns={'iso_a3': 'iso3'}), on='iso3', how='inner'))
    if len(data_) != len(data_):
        print(f'{len(data_) - len(data_)} countries were not found in the world map. They will be excluded from the plot.')
    data_.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Prepare variables and colormap
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
        world.boundary.plot(ax=m_ax, fc='lightgrey', lw=0, zorder=0, ec='k')
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

        if run_ols:
            add_regression(s_ax, data_, scatter_x_var, variable)

        sns_plot = sns.scatterplot(data=data_, x=scatter_x_var, y=variable, ax=s_ax,
                                   hue='Income group', hue_order=list(rename_income_groups.values()), alpha=.5,
                                   palette=INCOME_GROUP_COLORS,
                                   style='Income group', markers=INCOME_GROUP_MARKERS, s=10,
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

        # Histogram for the distribution of 'Income group'
        # sns.kdeplot(data=data_, y=variable, ax=hist_y_axs[i], legend=False, fill=True, color='k', alpha=.25, common_norm=True)
        # sns.histplot(data=data_, y=variable, hue='Income group', ax=hist_y_axs[i], legend=False, element='poly',
        #             palette=INCOME_GROUP_COLORS)
        sns.kdeplot(data=data_, y=variable, hue='Income group', ax=hist_y_axs[i], legend=False, fill=False,
                    common_norm=True,
                    palette=INCOME_GROUP_COLORS, alpha=.5)
        hist_y_axs[i].set_ylim(s_ax.get_ylim())

        hist_y_axs[i].axis('off')

        if i == 0 and not log_xaxis:
            # sns.kdeplot(data=data_, x='gdp_pc_pp', ax=hist_x_ax, legend=False, fill=True, common_norm=True, lw=0, alpha=.25, color='k')
            sns.kdeplot(data=data_, x='gdp_pc_pp', hue='Income group', ax=hist_x_ax, legend=False, fill=False, common_norm=True,
                        palette=INCOME_GROUP_COLORS, alpha=.5)
            hist_x_ax.set_xlim(s_ax.get_xlim())

            hist_x_ax.axis('off')

        # s_ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        s_ax.set_title('')
        s_ax.set_xlabel('')
        s_ax.set_ylabel('')

    for ax in plt_axs[:-1, 1]:
        ax.set_xticklabels([])

    plt_axs[-1, 1].set_xlabel(name_dict_[scatter_x_var])

    plt.tight_layout(pad=0)

    # add space between the subplots
    plt.subplots_adjust(hspace=hspace_adjust)

    for i, ((m_ax, s_ax), variable) in enumerate(zip(plt_axs, variables)):
        fig.text(0, .5, name_dict_[variable], transform=blended_transform_factory(fig.transFigure, m_ax.transAxes),
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
        plt.savefig(outfile, dpi=300, bbox_inches='tight', transparent=True, pad_inches=0)
    if show:
        plt.show(block=False)
        return
    else:
        plt.close()
        return plt_axs


def plot_fig_2(results_data_, income_cat_data_, outfile=None, show=False, numbering=True):
    fig_width = single_col_width * centimeter
    fig_heigt = 5.2 * centimeter
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(fig_width, fig_heigt),
                            gridspec_kw={'height_ratios': [1, 4], 'hspace': 0},
                            )
    ax_scatter = axs[1, 0]
    ax_kde = axs[0, 0]
    ax_boxplots = axs[1, 1]
    axs[0, 1].set_visible(False)
    # Scatter plot
    sns_scatter = sns.scatterplot(data=results_data_, x='gini_index', y='resilience', hue='Income group', style='Income group',
                                  palette=INCOME_GROUP_COLORS, ax=ax_scatter, legend=True, markers=INCOME_GROUP_MARKERS, alpha=.5, s=10,
                                  hue_order=list(rename_income_groups.values()))
    sns.move_legend(sns_scatter, 'upper right', bbox_to_anchor=(1, 1), frameon=False, title=None, handletextpad=-.25)
    # Adjust legend markers to have full opacity
    for legend_handle in sns_scatter.legend_.legend_handles:
        legend_handle.set_alpha(1)
    ax_scatter.set_xlabel('Gini index [%]')
    ax_scatter.set_ylabel('Resilience')

    add_regression(ax_scatter, results_data_, 'gini_index', 'resilience', p_val_pos='lower left')

    # KDE plot
    sns.kdeplot(data=results_data_, x='gini_index', hue='Income group', palette=INCOME_GROUP_COLORS, ax=ax_kde, fill=False, common_norm=False, legend=False, alpha=0.5)
    ax_kde.set_xlim(ax_scatter.get_xlim())
    ax_kde.set_xlabel('')
    ax_kde.set_ylabel('')
    ax_kde.set_xticklabels([])
    ax_kde.set_yticklabels([])
    ax_kde.set_xticks([])
    ax_kde.set_yticks([])
    ax_kde.axis('off')

    boxplot_data = income_cat_data_[['dw_rel', 'dk_rel']].stack() * 100
    boxplot_data.index.names = ['iso3', 'income quintile', 'loss_type']
    boxplot_data.name = 'shares [%]'
    boxplot_data = boxplot_data.reset_index()
    boxplot_data = boxplot_data.replace({'dw_rel': 'utility loss', 'dk_rel': 'asset loss'})

    sns_boxplot = sns.boxplot(x='income quintile', y='shares [%]', hue='loss_type',
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
        plt.show()
    else:
        plt.close()




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


def aggregate_to_quintile_data(cat_info_):
    quintile_data = cat_info_[['c']].droplevel(['hazard', 'rp', 'affected_cat', 'helped_cat']).drop_duplicates()
    for variable in ['dk', 'dw']:
        var_data = agg_to_event_level(cat_info_, variable, ['iso3', 'hazard', 'rp', 'income_cat'])
        var_data = average_over_rp(var_data, 'default_rp').groupby(['iso3', 'income_cat']).sum()
        quintile_data[variable] = var_data
        quintile_data[f'{variable}_rel'] = var_data / var_data.groupby('iso3').sum()
    # quintile_data['dw_pc_currency'] = quintile_data['dw'] / quintile_data.c.groupby('iso3').mean() ** (-1.5)
    # quintile_data['resilience'] = quintile_data['dk'] / quintile_data['dw_pc_currency']
    # quintile_data['dw_over_dk'] = quintile_data['dw_pc_currency'] / quintile_data['dk']
    return quintile_data


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
                  consumption_offset_=None, title=None, ylims=None, plot_legend=True, show_ylabel=True):
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
                        edgecolor='grey', label='Liquidity and PDS')
    axs[0].plot([-0.03 * (max(t_) - min(t_)), 0], [c_baseline, c_baseline], color='black', label='__none__')
    axs[0].plot([0, 0], [c_baseline, (c_baseline - di_h - dc_reco + dc_savings_pds)[0]], color='black', label='__none__')
    axs[0].plot(t_, c_baseline - di_h - dc_reco + dc_savings_pds, color='black', label='Consumption')

    dk_eff = delta_k_h_eff_of_t(t_, 0, delta_k_h_eff_, lambda_h_, sigma_h_, delta_c_h_max_, productivity_pi_)
    #axs[1].fill_between(t_, 0, dk_eff, color='red', alpha=0.5, label='Effective capital loss')
    axs[1].plot([-0.03 * (max(t_) - min(t_)), 0], [0, 0], color='black', label='__none__')
    axs[1].plot([0, 0], [0, dk_eff[0]], color='black', label='__none__')
    axs[1].plot(t_, dk_eff, color='black', label='Effective capital loss')

    axs[1].set_xlabel('Time [y]')

    if show_ylabel:
        axs[0].set_ylabel('Consumption\n(2021 PPP USD)')
        axs[1].set_ylabel('Capital loss\n(2021 PPP USD)')
    else:
        axs[0].set_ylabel(None)
        axs[1].set_ylabel(None)
        axs[0].set_yticklabels([])
        axs[1].set_yticklabels([])

    if ylims is not None:
        axs[0].set_ylim(ylims[0])
        axs[1].set_ylim(ylims[1])

    if title is not None:
        axs[0].set_title(title)
    if plot_legend:
        for ax in axs:
            ax.legend(frameon=False, bbox_to_anchor=(1, 1), loc='upper left')
    plt.tight_layout()


def plot_capital_shares(root_dir_, any_to_wb_, gdp_pc_, reconstruction_capital_='prv', outpath_=None):
    capital_shares = calc_reconstruction_share_sigma(root_dir_, any_to_wb_, reconstruction_capital_=reconstruction_capital_)
    capital_shares *= 100
    fig, axs = plt.subplots(ncols=3, figsize=(12, 4.5), sharex=True, sharey=True)
    capital_shares = pd.concat([capital_shares, gdp_pc_.rename('gdp_pc_pp')], axis=1)
    gdp = capital_shares.gdp_pc_pp / 1000
    for ax, (x, y), name in zip(axs, [('gdp_pc_pp', 'k_pub_share'), ('gdp_pc_pp', 'k_prv_share'),
                                      ('gdp_pc_pp', 'k_oth_share')],
                                [r'$\kappa^{public}$', r'$\kappa^{households}$', r'$\kappa^{firms}$']):
        if x == 'gdp_pc_pp':
            x_ = gdp
        else:
            x_ = capital_shares[x]
        ax.scatter(x_, capital_shares[y], marker='o')
        for i, label in enumerate(capital_shares.index):
            ax.text(x_[i], capital_shares[y][i], label, fontsize=10)
            ax.set_xlabel('GDP per capita / $1,000')
            ax.set_title(name)
        axs[0].set_ylabel('share (%)')
    plt.tight_layout()
    if outpath_ is not None:
        fig.savefig(os.path.join(outpath_, f"capital_shares_over_gpd.pdf"), dpi=300, bbox_inches='tight')
    plt.show(block=False)

    fig, axs = plt.subplots(ncols=3, figsize=(12, 4.5), sharex=True, sharey=True)
    for ax, (x, y), name in zip(axs, [('self_employment', 'k_pub_share'),
                                      ('self_employment', 'k_prv_share'),
                                      ('self_employment', 'k_oth_share')],
                                [r'$\kappa^{public}$', r'$\kappa^{households}$', r'$\kappa^{firms}$']):
        x_ = capital_shares[x]
        ax.scatter(x_, capital_shares[y], marker='o')
        for i, label in enumerate(capital_shares.index):
            ax.text(x_[i], capital_shares[y][i], label, fontsize=10)
            ax.set_xlabel('self employment rate (%)')
            ax.set_title(name)
        axs[0].set_ylabel('share (%)')
    plt.tight_layout()
    if outpath_ is not None:
        fig.savefig(os.path.join(outpath_, f"capital_shares_over_self_employment.pdf"), dpi=300, bbox_inches='tight')
    plt.show(block=False)

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    sns.scatterplot(data=capital_shares, x='gdp_pc_pp', y='self_employment', alpha=.5)
    ax.set_xlabel('GDP per capita (PPP USD)')
    ax.set_ylabel('self employment rate (%)')
    plt.tight_layout()
    if outpath_ is not None:
        fig.savefig(os.path.join(outpath_, f"self_employment_over_gdp.pdf"), dpi=300, bbox_inches='tight')
    plt.show(block=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script parameters')
    parser.add_argument('--climate_scenario', type=str, default='Existing climate')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--root_dir', type=str, default='./')
    args = parser.parse_args()

    climate_scenario = args.climate_scenario.replace(' ', '_')
    outpath = f"./figures/"
    os.makedirs(outpath, exist_ok=True)
    root_dir = args.root_dir

    simulation_paths = {
        'baseline': "2024-11-06_16-12_baseline_EW-2018",
    }

    results_data = {
        k: pd.read_csv(f"./output/scenarios/Existing_climate/{v}/results.csv") for k, v in simulation_paths.items()
    }

    cat_info_data = {
        k: pd.read_csv(f"./output/scenarios/Existing_climate/{v}/iah.csv", index_col=[0, 1, 2, 3, 4, 5]) for k, v in simulation_paths.items()
    }

    quintile_data = {
        k: aggregate_to_quintile_data(cat_info_data[k]) for k in cat_info_data.keys()
    }

    gini_index = get_wb_series('SI.POV.GINI').rename('gini_index').dropna().reset_index()
    gini_index = gini_index.loc[gini_index.groupby('country').year.idxmax()].drop(columns='year')
    gini_index = df_to_iso3(gini_index, 'country').set_index('iso3').drop(columns='country').squeeze()

    income_groups = load_income_groups()
    rename_income_groups = {
        'Low income': 'L',
        'Lower middle income': 'LM',
        'Upper middle income': 'UM',
        'High income': 'H'
    }

    for k in results_data.keys():
        results_data[k] = results_data[k][results_data[k].iso3 != 'THA']
        results_data[k][['resilience', 'risk', 'risk_to_assets']] *= 100
        results_data[k] = results_data[k].join(gini_index, on='iso3')
        results_data[k]['log_gdp_pc_pp'] = np.log(results_data[k]['gdp_pc_pp'])
        results_data[k]['gdp_pc_pp'] /= 1e3
        results_data[k] = pd.merge(results_data[k], income_groups, left_on='iso3', right_index=True, how='left')
        results_data[k]['Income group'] = results_data[k]['Income group'].replace(rename_income_groups)

    # read WB data
    wb_data_macro = load_input_data(root_dir, "WB_socio_economic_data/wb_data_macro.csv").set_index('iso3')
    wb_data_cat_info = load_input_data(root_dir, "WB_socio_economic_data/wb_data_cat_info.csv").set_index(
        ['iso3', 'income_cat'])

    name_dict = {
        'resilience': 'socio-economic resilience [%]',
        'risk': 'wellbeing losses [% of GDP]',
        'risk_to_assets': 'asset losses [% of GDP]',
        'gdp_pc_pp': 'GDP per capita [PPP USD]',
        'dk_tot': 'Asset losses [PPP USD]',
        'dWtot_currency': 'Welfare losses [PPP USD]',
        'gini_index': 'Gini index [%]',
    }
    any_to_wb, iso3_to_wb, iso2_iso3 = get_country_name_dicts("./")

    if args.plot:
        # plot capital shares
        plot_capital_shares(root_dir, any_to_wb, wb_data_macro.gdp_pc_pp, reconstruction_capital_='prv', outpath_=outpath)

        # plot recovery of ('LBN', 'Earthquake', 5000, 'q1', 'a', 'not_helped') for scenarios without PDS and savings and with
        household = ['LBN', 'Earthquake', 5000, 'q1', 'a']
        cat_info = pd.read_csv(f"./output/scenarios/Existing_climate/2024-05-15_11-32_no_liquidity_EW-2018_noPDS/iah_tax_no.csv", index_col=[0, 1, 2, 3, 4, 5])
        macro = pd.read_csv(f"./output/scenarios/Existing_climate/2024-05-15_11-32_no_liquidity_EW-2018_noPDS/macro_tax_no.csv", index_col=[0, 1, 2])
        data = pd.merge(macro, cat_info, left_index=True, right_index=True).loc[tuple(household + ['not_helped'])]
        data.recovery_params = [(float(d.split(', ')[0]), float(d.split(', ')[1])) for d in data.recovery_params[2:-2].split('), (')]
        plot_recovery(6, data.avg_prod_k, data.tau_tax, data.k, data.dk,
                      data.lambda_h, data.reconstruction_share_sigma_h, data.liquidity, data.help_received, np.nan,
                      data.recovery_params, data.gamma_SP * data.n, data.diversified_share,
                      ylims=[(-2000, 5500), None],
                      title=f"without liquidity and PDS")
        plt.savefig(f"./figures/{climate_scenario}/{household[0]}_{household[1]}_{household[2]}_{household[3]}_a_without_liqudity-PDS.pdf", dpi=300, bbox_inches='tight')
        cat_info = pd.read_csv(f"./output/scenarios/Existing_climate/2024-05-14_15-20_baseline_EW-2018/iah_tax_unif_poor.csv", index_col=[0, 1, 2, 3, 4, 5])
        macro = pd.read_csv(f"./output/scenarios/Existing_climate/2024-05-14_15-20_baseline_EW-2018/macro_tax_unif_poor.csv", index_col=[0, 1, 2])
        data = pd.merge(macro, cat_info, left_index=True, right_index=True).loc[tuple(household + ['helped'])]
        data.recovery_params = [(float(d.split(', ')[0]), float(d.split(', ')[1])) for d in data.recovery_params[2:-2].split('), (')]
        plot_recovery(6, data.avg_prod_k, data.tau_tax, data.k, data.dk,
                      data.lambda_h, data.reconstruction_share_sigma_h, data.liquidity, data.help_received, np.nan,
                      data.recovery_params, data.gamma_SP * data.n, data.diversified_share,
                      ylims=[(-2000, 5500), None],
                      title=f"with liquidity and PDS")
        plt.savefig(f"./figures/{climate_scenario}/{household[0]}_{household[1]}_{household[2]}_{household[3]}_a_with_liqudity-PDS.pdf", dpi=300, bbox_inches='tight')
        if False:
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
                data = data.loc[('COL', 'Earthquake', 5000, 'q1', 'a', 'helped')]
                plot_recovery(10, data.avg_prod_k, data.tau_tax, data.k, data.dk,
                              data.lambda_h, data.reconstruction_share_sigma_h, data.liquidity, data.help_received, np.nan,
                              data.recovery_params, data.gamma_SP * data.n, data.diversified_share,
                              ylims=[(-2200, 3000), None],
                              title=f"{'with' if with_savings else 'without'} liquid savings and borrowing")
                plt.savefig(
                    f"./figures/{climate_scenario}/COL_Earthquake_5000_q1_a_helped_{'with' if with_savings else 'without'}-liquidity.pdf",
                    dpi=300, bbox_inches='tight')

        plot_fig_1(
            data_=results_data['baseline'],
            bins_list={'resilience': [50, 60, 70, 80, 90, 100], 'risk': [0, .25, .5, 1, 2, 6],
                       'risk_to_assets': [0, .125, .25, .5, 1, 3]},
            cmap={'resilience': 'Reds_r', 'risk': 'Reds', 'risk_to_assets': 'Reds'},
            annotate=['HTI', 'TJK'],
            outfile=f"{outpath}/fig_1.pdf",
            log_xaxis=True,
            run_ols=True,
        )

        plot_fig_2(
            results_data_=results_data['baseline'],
            income_cat_data_=quintile_data['baseline'],
            outfile=f"{outpath}/fig_2.pdf",
            numbering=True
        )

        plot_drivers_gdp_and_gini_index(
            results_data['baseline'],
            outpath=outpath,
            annotate=['HTI', 'LAO', 'HND', 'TJK', 'GRC', 'MMR', 'URK', 'ECU', 'BTN', 'IRL', 'LUX', 'UKR', 'IRN', 'GEO']
        )

        make_income_cat_boxplots(
            results_path="./output/scenarios/Existing_climate/2024-05-14_15-20_baseline_EW-2018",
            outpath=outpath,
            focus_countries=['FIN', 'BOL', 'NAM'],
        )

        plot_hbar(
            data=results_data['baseline'],
            comparison_data=results_data['poor_reduction_results'],
            variables=["dWtot_currency", "dk_tot"],
            how='abs',
            unit='millions',
            name_dict=name_dict,
            precision=0,
            outfile=f"{outpath}/poor_exposure_reduction_dW-dk_comparison_abs.pdf",
        )
        plot_hbar(
            data=results_data['baseline'],
            comparison_data=results_data['poor_reduction_results'],
            variables=["dWtot_currency", "dk_tot"],
            how='rel',
            name_dict=name_dict,
            outfile=f"{outpath}/poor_exposure_reduction_dW-dk_comparison_rel.pdf",
        )
        plot_hbar(
            # data=datasets['baseline'][datasets['baseline'].iso3.isin(['CHN', 'IND', 'IDN', 'USA', 'ITA', 'TUR', 'JPN', 'VNM', 'IRN', 'PAK', 'GRC', 'KOR', 'COL', 'PHL', 'URK'])],
            data=results_data['baseline'],
            comparison_data=results_data['nonpoor_reduction_results'],
            variables=["dWtot_currency", "dk_tot"],
            how='abs',
            unit='millions',
            name_dict=name_dict,
            precision=0,
            outfile=f"{outpath}/nonpoor_exposure_reduction_dW-dk_comparison_abs.pdf",
        )
        plot_hbar(
            # data=datasets['baseline'][datasets['baseline'].iso3.isin(['AZE', 'NGA', 'ALB', 'SVK', 'IDN', 'NPL', 'HND', 'CYP', 'GHA', 'HTI', 'LAO', 'COL', 'AGO', 'PAK', 'IND'])],
            data=results_data['baseline'],
            comparison_data=results_data['nonpoor_reduction_results'],
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

