import sys
import os
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scenario.data.get_wb_data import get_wb_mrv
import argparse
import os
import string
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import BoundaryNorm
import matplotlib.colors as mcolors
from matplotlib.transforms import blended_transform_factory
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle, Polygon
import statsmodels.api as sm
import numpy as np
import pandas as pd
import geopandas as gpd
import cartopy.crs as ccrs
from misc.helpers import average_over_rp, get_country_name_dicts, load_income_groups, df_to_iso3
from model.recovery_optimizer import baseline_consumption_c_h, delta_c_h_of_t, delta_k_h_eff_of_t
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


def load_data(simulation_paths_, model_root_dir_):
    gini_index_ = df_to_iso3(get_wb_mrv('SI.POV.GINI', 'gini_index').reset_index(), 'country')
    gini_index_ = gini_index_.set_index('iso3').drop('country', axis=1).squeeze()

    income_groups_ = load_income_groups(os.path.join(model_root_dir_, 'scenario'))

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
        if 'THA' in results_data_[k].index:
            results_data_[k] = results_data_[k].drop('THA')
        results_data_[k][['resilience', 'risk', 'risk_to_assets']] *= 100
        results_data_[k] = results_data_[k].join(gini_index_, on='iso3')
        results_data_[k]['log_gdp_pc_pp'] = np.log(results_data_[k]['gdp_pc_pp'])
        results_data_[k] = pd.merge(results_data_[k], income_groups_, left_on='iso3', right_index=True, how='left')

    hazard_protection_ = {
        k: pd.read_csv(os.path.join(v, "scenario__hazard_protection.csv"), index_col=[0, 1]) for k, v in
        simulation_paths_.items()
    }

    name_dict_ = {
        'resilience': 'socio-economic resilience [%]',
        'risk': 'risk to wellbeing [% of GDP]',
        'risk_to_assets': 'risk to assets [% of GDP]',
        'gdp_pc_pp': 'GDP per capita [$PPP]',
        'dk_tot': 'Asset losses [$PPP]',
        'dWtot_currency': 'Welfare losses [$PPP]',
        'gini_index': 'Gini index [%]',
    }
    any_to_wb_, iso3_to_wb_, iso2_iso3_ = get_country_name_dicts(os.path.join(model_root_dir_, 'scenario'))

    # financial preparedness and preparedness to scale up the support are not used in insurance / PDS simulations
    data_coverage_ = pd.read_csv(os.path.join(simulation_paths_['baseline'], 'data_coverage.csv'), index_col=0)
    data_coverage_.drop(['finance_pre', 'prepare_scaleup', 'borrowing_ability'], axis=1, inplace=True)
    data_coverage_.rename({
            'gdp_pc_pp': 'per capita GDP',
            'pop': 'Population',
            'income_share': 'Income share held by each income quintile',
            'transfers': 'Fraction of income from social protection and transfers',
            'findex': 'Liquidity and financial inclusion',
            'ew': 'Availability of early warning systems',
            'avg_prod_k': 'Average productivity of capital',
            'k_pub_share': 'Public capital share',
            'self_employment': 'Self-employment rate',
            'real_estate_share_of_value_added': 'Real estate share of value added',
            'home_ownership_rate': 'Home ownership rate',
            'hazard_loss': 'Asset losses',
            'gem_building_classes': 'Building inventory',
            'gmd_vulnerability_rel': 'Dwelling materials microdata',
            'exposure_bias': 'Number of exposed people by hazard type and poverty line',
            'flopros': 'Flood protection levels',
        }, axis=1, inplace=True)

    return income_groups_, gini_index_, cat_info_data_, macro_data_, results_data_, hazard_protection_, name_dict_, any_to_wb_, iso3_to_wb_, iso2_iso3_, data_coverage_


def print_stats(results_data_):
    # print the asset losses, wellbeing losses, and resilience for TJK and HTI
    for c in ['HTI', 'TJK']:
        print(
            f'{c}: dK={results_data_.loc[c, "dk_tot"] / 1e9}, dC^eq={results_data_.loc[c, "dWtot_currency"] / 1e9}, Psi={results_data_.loc[c, "resilience"]}')

    # print the population-weighted average resilience
    print(
        f'Global: dK={results_data_.dk_tot.sum() / 1e9}, dC^eq={results_data_.dWtot_currency.sum() / 1e9}, resilience={results_data_.dk_tot.sum() / results_data_.dWtot_currency.sum()}')

    # print the stats of the resilience by Country income group
    print('Resilience by Country income group:')
    print(results_data_.groupby('Country income group').resilience.describe().loc[['LICs', 'LMICs', 'UMICs', 'HICs']])

    # print minimum and maximum resilience of each Country income group
    print('Minimum and maximum resilience of each Country income group:')
    print(results_data_.groupby('Country income group').resilience.agg(['min', 'max', 'idxmin', 'idxmax']))


def print_results_table(results_data_, data_coverage_=None):
    if data_coverage_ is not None:
        superscript_columns = data_coverage.columns[(data_coverage != 'available').any(axis=0)]
        superscript_letters = np.array([chr(97 + i) for i in range(len(superscript_columns))])
        print(f"Superscripts:\n" , list(zip(superscript_letters, superscript_columns)))
    for idx, row in results_data_.iterrows():
        superscripts = ''
        if data_coverage_ is not None:
            superscripts = '\\textsuperscript{' + ','.join(superscript_letters[data_coverage_.loc[idx, superscript_columns] != 'available']) + '}'
            if superscripts == '\\textsuperscript{}':
                superscripts = ''
        print(f'{idx}{superscripts} & {row["Country income group"]} & {row["gdp_pc_pp"]:.0f} & {row["risk_to_assets"]:.2f} & {row["risk"]:.2f}  & {row["resilience"]:.2f}\\\\')


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
                                                                sigma_h_, savings_s_h_, delta_i_h_pds_,
                                                                recovery_params_, social_protection_share_gamma_h_,
                                                                consumption_floor_xi_, t_hat_, consumption_offset_,
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

    dk_eff = delta_k_h_eff_of_t(t_, delta_k_h_eff_, lambda_h_)

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


def plot_supfig_10(results_data_, data_coverage_, world_, show=True, outpath_=None, numbering_=True):
    map_vars = data_coverage.columns[~(data_coverage == 'available').all(axis=0)]
    num_maps = len(map_vars)
    fig = plt.figure(figsize=(double_col_width * centimeter, 0.25 * np.ceil(num_maps / 2) * double_col_width * centimeter))
    proj = ccrs.Robinson(central_longitude=0, globe=None)
    axs = [fig.add_subplot(int(np.ceil(num_maps / 2)), 2, i + 1, projection=proj) for i in range(num_maps)]
    for var, ax in zip(map_vars, axs):
        ax.set_extent([-180, 180, -60, 90])
        ax.set_rasterized(True)

        imputed_countries = data_coverage_[data_coverage_[var] == 'imputed'][var].index.values
        available_countries = data_coverage_[data_coverage_[var] == 'available'][var].index.values
        missing_countries = data_coverage_[data_coverage_[var].isna()][var].index.values

        world_.drop(results_data_.index).boundary.plot(ax=ax, fc='lightgrey', zorder=0, lw=0)

        for countries, c in zip([imputed_countries, available_countries, missing_countries], ['purple', 'green', 'dimgrey']):
            if len(countries) > 0:
                world_.loc[countries].boundary.plot(ax=ax, fc=c, zorder=5, lw=0)

        ax.set_title(var)

    plt.tight_layout(h_pad=2, w_pad=1.08)
    if numbering_:
        for i, ax in enumerate(axs):
            ax.text(i%2 * .5, 1, f'{chr(97 + i)}', ha='left', va='top', fontsize=8, fontweight='bold',
                    transform=blended_transform_factory(fig.transFigure, ax.transAxes))
    if show:
        plt.show(block=False)
    if outpath_:
        plt.savefig(outpath_, dpi=300, bbox_inches='tight')


def plot_supfig_9(cat_info_data_, outfile=None, show=True, numbering=True):
    # select rp=10 because liquidity and dk_reco are constant over rp
    plot_data = cat_info_data_.loc[pd.IndexSlice[:, :, 10, :, 'a', 'not_helped'], 'v_ew']
    plot_data = plot_data.droplevel(['affected_cat', 'helped_cat', 'rp'])

    # v_rel = (plot_data / plot_data.groupby(['iso3', 'hazard']).mean()).rename('v_rel').to_frame()
    v_rel = plot_data.rename('v_rel')

    v_rel = pd.merge(
        v_rel, income_groups['Country income group'], left_index=True, right_index=True
    )

    fig_width = double_col_width * centimeter
    fig_heigt = fig_width * .75
    fig, axs = plt.subplots(figsize=(fig_width, fig_heigt), ncols=3, nrows=2, sharey=True, sharex=False)
    axs.flatten()[-1].remove()

    # for cig, ax in zip(['LICs', 'LMICs', 'UMICs', 'HICs'], axs):
    for idx, (hazard, ax) in enumerate(zip(plot_data.index.get_level_values('hazard').unique(), axs.flatten())):
        legend = idx == 2
        # sns.boxplot(data=v_rel[v_rel['Country income group'] == cig], x='income_cat', y='v_rel', hue='hazard', ax=ax,
        #             legend=legend, showfliers=False)
        sns.boxplot(data=v_rel.xs(hazard, level='hazard'), x='income_cat', y='v_rel', hue='Country income group', ax=ax,
                    legend=legend, showfliers=False, hue_order=['LICs', 'LMICs', 'UMICs', 'HICs'])
        if legend:
            ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(1, 1))
        ax.set_ylabel('Household vulnerability')
        ax.set_xlabel('Household income quintile')
        ax.set_title(hazard)

    plt.tight_layout(h_pad=1.08)

    if numbering:
        for i, ax in enumerate(axs.flatten()):
            ax.text(-.05, 1.08, f'{chr(97 + i)}', ha='left', va='top', fontsize=8, fontweight='bold',
                    transform=ax.transAxes)

    if outfile:
        plt.savefig(outfile, dpi=300, bbox_inches='tight')

    if show:
        plt.show(block=False)


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



def plot_supfig_7(cat_info_data_, outfile=None, numbering=True, show=True):
    # select rp=10 because liquidity and dk_reco are constant over rp
    plot_data = cat_info_data_.loc[pd.IndexSlice[:, :, 10, :, 'a', 'not_helped'], ['dk_reco', 'liquidity']]
    plot_data = plot_data.droplevel(['affected_cat', 'helped_cat', 'rp'])

    liquidity_over_dk_reco = (plot_data.liquidity / plot_data.dk_reco).rename('liquidity_over_dk_reco').to_frame()
    liquidity_rel = plot_data.xs('Earthquake', level='hazard').liquidity / plot_data.xs('Earthquake', level='hazard').groupby('iso3').liquidity.mean()
    liquidity_rel = liquidity_rel.rename('liquidity').to_frame()

    fig_width = double_col_width * centimeter
    fig_heigt = fig_width / 2
    fig, axs = plt.subplots(figsize=(fig_width, fig_heigt), ncols=2)

    sns.boxplot(data=liquidity_over_dk_reco, x='income_cat', y='liquidity_over_dk_reco', hue='hazard', ax=axs[0])
    axs[0].legend(frameon=False)
    sns.boxplot(data=liquidity_rel, x='income_cat', y='liquidity', ax=axs[1])

    for ax in axs:
        ax.set_xlabel('Household income quintile')
    axs[0].set_ylabel('Liquidity relative to reconstruction cost')
    axs[1].set_ylabel('Liquidity relative to average liquidity')

    plt.tight_layout(h_pad=1.08)

    if numbering:
        for i, ax in enumerate(axs):
            ax.text(-.15, 1, f'{chr(97 + i)}', ha='left', va='top', fontsize=8, fontweight='bold', transform=ax.transAxes)

    if outfile:
        plt.savefig(outfile, dpi=300, bbox_inches='tight')

    if show:
        plt.show(block=False)


def plot_supfig_6(results_data_, cat_info_data_, plot_rp=None, outfile=None, show=False):
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


def plot_supfig_5(results_data_, outfile=None, show=False):
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


def plot_supfig_4(results_data_, outfile=None, show=False):
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


def plot_supfig_3(results_data_, outpath_=None):
    capital_shares = results_data_.copy()
    fig, axs = plt.subplots(ncols=3, nrows=3, figsize=(double_col_width * centimeter, 16 * centimeter), sharex=False, sharey='row')
    capital_shares[['k_priv_share', 'k_household_share', 'owner_occupied_share_of_value_added', 'self_employment']] *= 100
    capital_shares['k_pub_share'] = 100 - capital_shares['k_priv_share'] - capital_shares['k_household_share']
    capital_shares['gdp_pc_pp'] /= 1e3

    for ax, (x, y), name in zip(axs[0, :], [('gdp_pc_pp', 'k_pub_share'), ('gdp_pc_pp', 'k_priv_share'), ('gdp_pc_pp', 'k_household_share')], [r'$\kappa^p$', r'$\kappa^f$', r'$\kappa^h$']):
        legend = False
        if ax == axs[0, -1]:
            legend = True
        sns.scatterplot(capital_shares, x=x, y=y, ax=ax, alpha=.5, s=10, hue='Country income group', hue_order=['LICs', 'LMICs', 'UMICs', 'HICs'], legend=legend, palette=INCOME_GROUP_COLORS,
                                   style='Country income group', markers=INCOME_GROUP_MARKERS)
        for label in capital_shares.index:
            ax.text(capital_shares[x].loc[label], capital_shares[y].loc[label], label, fontsize=6)
        ax.set_xlabel('GDP per capita [$PPP 1,000]')
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

    for ax, (x, y), name in zip(axs[2, :], [('owner_occupied_share_of_value_added', 'k_pub_share'), ('owner_occupied_share_of_value_added', 'k_priv_share'), ('owner_occupied_share_of_value_added', 'k_household_share')], [r'$\kappa^p$', r'$\kappa^f$', r'$\kappa^h$']):
        x_ = capital_shares[x]
        sns.scatterplot(capital_shares, x=x, y=y, ax=ax, alpha=.5, s=10, hue='Country income group', hue_order=['LICs', 'LMICs', 'UMICs', 'HICs'], legend=False, palette=INCOME_GROUP_COLORS,
                                   style='Country income group', markers=INCOME_GROUP_MARKERS)
        for label in capital_shares.index:
            ax.text(x_.loc[label], capital_shares[y].loc[label], label, fontsize=6)
            ax.set_xlabel('Owner-occupied housing share\nof value added [%]')
        axs[2, 0].set_ylabel('share [%]')

    plt.tight_layout()
    if outpath_ is not None:
        fig.savefig(outpath_, dpi=300, bbox_inches='tight')
    plt.show(block=False)



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
            recovery_params_=q_data.recovery_params,
            social_protection_share_gamma_h_=q_data.gamma_SP,
            consumption_floor_xi_=None,
            t_hat=None,
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
        'pds40': '9: PDS equal to 40% of\n    asset losses of the poor',
        'insurance20': '10: National insurance covering\n      20% of all asset losses',
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

    bc_ratios = []
    for asp_variant in ['pds40', 'insurance20']:
        benefit = (results_data_['baseline'][['risk', 'gdp_pc_pp']].prod(axis=1) - results_data_[asp_variant][['risk', 'gdp_pc_pp']].prod(axis=1)) / 100
        cost = results_data[asp_variant]['average_aid_cost_pc']
        bc_ratios.append((benefit / cost).rename(asp_variant))
    returns_merged = pd.merge(pd.concat(bc_ratios, axis=1), income_groups, left_index=True, right_index=True)
    print(returns_merged.groupby('Country income group')[['pds40', 'insurance20']].describe())

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

    duration_ctry = gpd.GeoDataFrame(pd.merge(duration_ctry, world_, left_index=True, right_index=True, how='inner'))
    duration_ctry.plot(column='t_reco_avg', ax=axs[0], cax=axs[1], zorder=5, cmap=truncated_cmap, norm=norm, lw=0, legend=True, legend_kwds={'orientation': 'horizontal', 'shrink': 0.6, 'aspect': 30, 'fraction': .1, 'pad': 0})

    world_.drop(duration_ctry.index).boundary.plot(ax=axs[0], fc='lightgrey', lw=0, zorder=0, ec='k')
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


def plot_fig_3(results_data_, cat_info_data_, hazard_protection_, outfile=None, show=False, numbering=True):
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
        var_data = average_over_rp(var_data, hazard_protection_).groupby(['iso3', 'income_cat']).sum()
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


def plot_fig_2(data_, world, exclude_countries=None, bins_list=None, cmap='viridis', outfile=None,
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
    data_ = gpd.GeoDataFrame(pd.merge(data_, world, on='iso3', how='inner'))
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
        world.drop(data_.index).boundary.plot(ax=m_ax, fc='lightgrey', lw=0, zorder=0, ec='k')
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
            for txt, row in data_.loc[annotate].iterrows():
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
    axs[0].set_ylabel('Consumption\n[$PPP 1,000 / yr]')
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
    ax.set_title('Income\n[$PPP 1,000 / yr]')

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


# def compute_national_recovery_duration(cat_info_data_, outpath=None):
#     """
#     Compute the national recovery duration based on the recovery parameters
#     """
#     result = cat_info_data_.xs('a', level='affected_cat')[['k', 'dk', 'n']]
#     result['k_tot'] = result.k * result.n
#     result['dk_tot'] = result.dk * result.n
#     result = result.groupby(['iso3', 'hazard', 'rp']).dk_tot.sum() / result.groupby(['iso3', 'hazard', 'rp']).k_tot.sum() * 100
#     result = result.rename('capital_loss_rel').to_frame()
#
#     for idx in tqdm.tqdm(result.index, desc='Computing national recovery duration'):
#         idx_data = cat_info_data_.loc[idx].xs('a', level='affected_cat')
#         dk_tot = idx_data[['dk', 'n']].prod(axis=1)
#
#         if dk_tot.sum() == 0:
#             result.loc[idx, 't_reco_95'] = np.nan
#             continue
#
#         def fun(t, full_reco=.95):
#             # the function value
#             dk_of_t = dk_tot * np.exp(-idx_data.lambda_h * t) / dk_tot.sum()
#
#             # the derivative
#             ddk_of_t = -idx_data.lambda_h * dk_of_t
#
#             # the second derivative
#             d2dk_of_t = idx_data.lambda_h**2 * dk_of_t
#
#             return dk_of_t.sum() - (1 - full_reco), ddk_of_t.sum(), d2dk_of_t.sum()
#
#         # find the time to recover 95% of the asset losses
#         t_reco_95 = root_scalar(fun, fprime=True, fprime2=True, x0=0, bracket=(0, 1e5 )).root
#         result.loc[idx, 't_reco_95'] = t_reco_95
#         # result.loc[idx, 'probability'] = 1 / idx[2]
#
#     if outpath:
#         result.to_csv(outpath)
#     return result#, durations


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script parameters')
    parser.add_argument('simulation_outputs_dir', type=str)
    parser.add_argument('outpath', type=str)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()

    outpath = args.outpath
    os.makedirs(outpath, exist_ok=True)
    model_root_dir = os.path.dirname(Path(os.path.abspath(__file__)).parent)

    simulation_paths = {
        'baseline': '0_baseline',
        'reduce_total_exposure_0.05': '1_reduce_total_exposure/q1+q2+q3+q4+q5/0.95',
        'reduce_poor_exposure_0.05': '1_reduce_total_exposure/q1/0.95',
        'reduce_total_vulnerability_0.05': '3_reduce_total_vulnerability/q1+q2+q3+q4+q5/0.95',
        'reduce_poor_vulnerability_0.05': '3_reduce_total_vulnerability/q1/0.95',
        'increase_gdp_pc_and_liquidity_0.05': '5_scale_income_and_liquidity/q1+q2+q3+q4+q5/1.05',
        'reduce_self_employment_0.1': '6_scale_self_employment/q1+q2+q3+q4+q5/0.9',
        'reduce_non_diversified_income_0.1': '7_scale_non_diversified_income/q1+q2+q3+q4+q5/0.9',
        'pds40': '8_post_disaster_support/q1+q2+q3+q4+q5/0.4',
        'insurance20': '9_insurance/q1+q2+q3+q4+q5/0.2',
        'noLiquidity': '10_scale_income_and_liquidity/q1+q2+q3+q4+q5/0',
        'reduce_gini_10': '11_scale_gini_index/0.9',
    }
    simulation_paths = {k: os.path.join(args.simulation_outputs_dir, v) for k, v in simulation_paths.items()}

    income_groups, gini_index, cat_info_data, macro_data, results_data, hazard_protection, name_dict, any_to_wb, iso3_to_wb, iso2_iso3, data_coverage = load_data(simulation_paths, model_root_dir)

    gadm_world = gpd.read_file("/Users/robin/data/GADM/gadm_410-levels.gpkg", layer='ADM_0').set_crs(4326).to_crs('World_Robinson')
    gadm_world = gadm_world[~gadm_world.COUNTRY.isin(['Antarctica', 'Caspian Sea'])]
    gadm_world.rename(columns={'GID_0': 'iso3'}, inplace=True)
    gadm_world.set_index('iso3', inplace=True)

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
            world=gadm_world,
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
            hazard_protection_=hazard_protection['baseline'],
            outfile=f"{outpath}/fig_3.pdf",
            numbering=True,
            show=True,
        )

        plot_fig_4(
            cat_info_data_=cat_info_data['baseline'],
            income_groups_=income_groups,
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

        plot_supfig_3(
            results_data_=results_data['baseline'],
            outpath_=f"{outpath}/supfig_3.pdf",
        )

        plot_supfig_4(
            results_data_=results_data['baseline'],
            outfile=f"{outpath}/supfig_4.pdf",
            show=True,
        )

        plot_supfig_5(
            results_data_=results_data['baseline'],
            outfile=f"{outpath}/supfig_5.pdf",
            show=True,
        )

        plot_supfig_6(
            results_data_=results_data['baseline'],
            cat_info_data_=cat_info_data['baseline'],
            plot_rp=None,
            outfile=f"{outpath}/supfig_6.pdf",
            show=True,
        )

        plot_supfig_7(
            cat_info_data_=cat_info_data['baseline'],
            outfile=f"{outpath}/supfig_7.pdf",
            numbering=True,
            show=True,
        )

        plot_supfig_8(
            cat_info_data_=cat_info_data['noLiquidity'],
            outfile=f"{outpath}/supfig_8.pdf",
            show=True,
            numbering=False,
            plot_rp=None,
        )

        plot_supfig_9(
            cat_info_data_=cat_info_data['baseline'],
            outfile=f"{outpath}/supfig_9.pdf",
            show=True,
        )

        plot_supfig_10(
            results_data_=results_data['baseline'],
            world_=gadm_world,
            data_coverage_=data_coverage,
            outpath_=f"{outpath}/supfig_10.pdf",
            show=True,
        )

    print_stats(
        results_data_=results_data['baseline'],
    )

    print_results_table(
        results_data_=results_data['baseline'],
        data_coverage_=data_coverage,
    )
