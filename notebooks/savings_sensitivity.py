from matplotlib import pyplot as plt
import sys
sys.path.append('../')
from lib_compute_resilience_and_risk import *
import os
import warnings
import pandas as pd

warnings.filterwarnings("always", category=UserWarning)

if __name__ == '__main__':
    experiment_countries = ['MOZ', 'BGD', 'ISL']
    liquidity_factors = sorted(
        np.unique(list(np.arange(0, 3 + .1, 0.1)) + list(np.arange(.01, 0.101, 0.01)) + list(np.arange(3, 11, 1))))

    # define directory
    model = '/Users/robin/sync/git/global-unbreakable-model/'  # get current directory
    input_dir = model + '/inputs/'  # get inputs data directory
    intermediate_dir = model + '/intermediate/'  # get outputs data directory

    scenarios = os.listdir(os.path.join(intermediate_dir, 'scenarios'))
    scenarios = [s for s in scenarios if os.path.isdir(os.path.join(intermediate_dir, 'scenarios', s))]

    # TODO: remove this later; only for testing purposes
    scenarios = ['baseline']

    for scenario in scenarios:
        print(scenario)
        option_fee = "tax"
        option_pds = "unif_poor"

        if option_fee == "insurance_premium":
            option_b = 'unlimited'
            option_t = 'perfect'
        else:
            option_b = 'data'
            option_t = 'data'

        print(f'optionFee ={option_fee}, optionPDS ={option_pds}, optionB ={option_b}, optionT ={option_t}')

        # Options and parameters
        econ_scope = "iso3"  # province, deparmtent
        event_level = [econ_scope, "hazard", "rp"]  # levels of index at which one event happens
        default_rp = "default_rp"  # return period to use when no rp is provided (mind that this works with protection)
        affected_cats = pd.Index(["a", "na"], name="affected_cat")  # categories for social protection
        helped_cats = pd.Index(["helped", "not_helped"], name="helped_cat")
        poor_cat = 'q1'

        # read data

        # macro-economic country economic data
        macro = pd.read_csv(os.path.join(intermediate_dir, 'scenarios', scenario, "scenario__macro.csv"),
                            index_col=econ_scope)

        # consumption, access to finance, gamma, capital, exposure, early warning access by country and income category
        cat_info = pd.read_csv(os.path.join(intermediate_dir, 'scenarios', scenario, "scenario__cat_info.csv"),
                               index_col=[econ_scope, "income_cat"])

        # exposure, vulnerability, and access to early warning by country, hazard, return period, income category
        hazard_ratios = pd.read_csv(
            os.path.join(intermediate_dir, 'scenarios', scenario, "scenario__hazard_ratios.csv"),
            index_col=event_level + ["income_cat"])

        hazard_protection = pd.read_csv(
            os.path.join(intermediate_dir, 'scenarios', scenario, "scenario__hazard_protection.csv"),
            index_col=[econ_scope, "hazard"])

        # compute
        # reshape macro and cat_info to event level, move hazard_ratios data to cat_info_event
        macro_event, cat_info_event = reshape_input(
            macro=macro,
            cat_info=cat_info,
            hazard_ratios=hazard_ratios,
            event_level=event_level,
        )

        macro_event = macro_event.loc[experiment_countries]
        cat_info_event = cat_info_event.loc[experiment_countries]

        # calculate the potential damage to capital, and consumption
        # adds 'dk', 'dc', 'dc_npv_pre' to cat_info_event_ia, also adds aggregated 'dk' to macro_event
        macro_event, cat_info_event_ia = compute_dK(
            macro_event=macro_event,
            cat_info_event=cat_info_event,
            event_level=event_level,
            affected_cats=affected_cats,
        )

        # calculate the post-disaster response
        # adds 'error_incl', 'error_excl', 'max_aid', 'need', 'aid', 'unif_aid' to macro_event
        # adds 'help_received', 'help_fee', 'help_needed' to cat_info_event_ia(h)
        macro_event, cat_info_event_iah = calculate_response(
            macro_event=macro_event,
            cat_info_event_ia=cat_info_event_ia,
            event_level=event_level,
            poor_cat=poor_cat,
            helped_cats=helped_cats,
            option_fee=option_fee,
            option_t=option_t,
            option_pds=option_pds,
            option_b=option_b,
            loss_measure="dk",
            fraction_inside=1,
            share_insured=.25,
        )
        results = {}
        for f_idx, f in enumerate(liquidity_factors):
            print(f'\n#####\nliquidity factor: {f_idx + 1} of {len(liquidity_factors)}')
            cat_info_event_iah_, macro_event_ = compute_dw_new(
                cat_info_event_iah=cat_info_event_iah.assign(liquidity=f * cat_info_event_iah.liquidity),
                macro_event=macro_event,
                event_level_=event_level,
                capital_t=100,
                delta_c_h_max=np.nan,
            )

            # # compute welfare losses
            # # adds 'dc_npv_post', 'dw' to cat_info_event_iah
            # cat_info_event_iah = compute_dW(
            #     macro_event=macro_event,
            #     cat_info_event_iah_=cat_info_event_iah,
            # )

            # aggregate to event-level (ie no more income_cat, helped_cat, affected_cat, n)
            # Computes results
            # results consists of all variables from macro, as well as 'aid' and 'dk' from macro_event
            results[f] = prepare_output(
                macro=macro,
                macro_event=macro_event,
                cat_info_event_iah=cat_info_event_iah_,
                event_level=event_level,
                hazard_protection_=hazard_protection,
                econ_scope=econ_scope,
                default_rp=default_rp,
                is_local_welfare=True,
                return_stats=True,
            )

        fig, ax = plt.subplots(ncols=len(experiment_countries), nrows=3, figsize=(10, 8), sharex=True)
        for c_idx, (country, color) in enumerate(zip(experiment_countries, ['tab:blue', 'tab:orange', 'tab:red'])):
            dw_reco = [r_data.loc[country, 'dW_reco'] for r_data in results.values()]
            dw_long_term = [r_data.loc[country, 'dW_long_term'] for r_data in results.values()]
            resilience = [r_data.loc[country, 'resilience'] for r_data in results.values()]
            dS = [lf_next - lf_prev for lf_next, lf_prev in zip(liquidity_factors[1:], liquidity_factors[:-1])]
            dw_reco_slope = [(next_val - prev_val) / dS_ for next_val, prev_val, dS_ in zip(dw_reco[1:], dw_reco[:-1], dS)]
            dw_long_term_slope = [(next_val - prev_val) / dS_ for next_val, prev_val, dS_ in
                                  zip(dw_long_term[1:], dw_long_term[:-1], dS)]
            ax[0, c_idx].plot(liquidity_factors, dw_reco, label=r'$\Delta W^{reco}$', color=color,
                              linestyle='dotted')
            ax[0, c_idx].plot(liquidity_factors, dw_long_term, label=r'$\Delta W^{long-term}$',
                              color=color, linestyle='--')
            ax[0, c_idx].plot(liquidity_factors, [w1 + w2 for w1, w2 in zip(dw_long_term, dw_reco)], label=r'$\Delta W$',
                              color=color)
            ax[1, c_idx].plot(liquidity_factors, resilience, label='resilience', color=color)
            ax[2, c_idx].plot(liquidity_factors[1:], dw_reco_slope,
                              label=r'$\frac{d}{d S}\Delta W^{reco}$', color=color,
                              linestyle='dotted')
            ax[2, c_idx].plot(liquidity_factors[1:], dw_long_term_slope,
                              label=r'$\frac{d}{d S}\Delta W^{long-term}$', linestyle='--',
                              color=color)
            ax[2, c_idx].plot(liquidity_factors[1:], [s1 + s2 for s1, s2 in zip(dw_long_term_slope, dw_reco_slope)],
                              label=r'$\frac{d}{d S}\Delta W$', color=color)
            for ax_ in ax[:, -1].flatten():
                ax_.legend()
            for ax_ in ax[[0, 2], :].flatten():
                ax_.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            for ax_, ylabel in zip(ax[:, 0], [r'$\Delta W$', 'resilience', r'$\frac{d}{d S}\Delta W$']):
                ax_.set_ylabel(ylabel)
            for ax_ in ax[-1, :]:
                ax_.set_xlabel(r'$\frac{S}{S^{FINDEX}}$')
            for ax_ in ax.flatten():
                ax_.axvline(1, color='k', lw=.5)
            plt.tight_layout()
            plt.show(block=False)
            plt.draw()
