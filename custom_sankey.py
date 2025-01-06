import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



from plotting import plot_recovery
from recovery_optimizer import delta_c_h_of_t

# figure widths
# "Column-and-a-Half: 120–136 mm wide"
single_col_width = 8.9  # cm
double_col_width = 18.3  # cm
max_fig_height = 24.7  # cm

inch = 2.54
centimeter = 1 / inch


def plot_sankey_flow(cat_info_data_, macro_data_, iso3='HTI', hazard='Earthquake', plot_rp=100):
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
            sigma_h_=q_data.reconstruction_share_sigma_h,
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
        plot_data.loc[q, 'k'] = q_data.k
        plot_data.loc[q, 'dk'] = plot_data.loc[q, 'k'] * q_data.v_ew
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

    plot_data.loc['q_avg', ['k', 'y_bl', 'tr_bl', 'c_bl', 'dc_income_sp', 'c_income_sp']] = plot_data[['k', 'y_bl', 'tr_bl', 'c_bl', 'dc_income_sp', 'c_income_sp']].mean()
    plot_data.loc['q_avg', ['dk', 'dy', 'dtr', 'dc_savings', 'dc_reco', 'dc_income_lab', 'dw_short_term']] = 0
    plot_data.loc['q_avg', 'dc_income'] = plot_data.loc['q_avg', 'dc_income_sp']
    plot_data.loc['q_avg', 'c_income_lab'] = plot_data.loc['q_avg', 'y_bl'] * (1 - macro_data_.tau_tax.loc[iso3, hazard, plot_rp])
    plot_data.loc['q_avg', 'c_income'] = plot_data.loc['q_avg', 'c_income_lab'] + plot_data.loc['q_avg', 'c_income_sp']
    plot_data.loc['q_avg', 'dc_short_term'] = plot_data.loc['q_avg', 'dc_income_sp']
    plot_data.loc['q_avg', 'dc_long_term'] = (plot_data.dc_long_term - plot_data.dc_savings).mean()
    plot_data.loc['q_avg', ['y', 'tr', 'c']] = plot_data.loc['q_avg', ['y_bl', 'tr_bl', 'c_bl']].values - plot_data.loc['q_avg', ['dy', 'dtr', 'dc_short_term']].values
    plot_data.loc['q_avg', ['dw', 'dw_long_term']] = data.loc[pd.IndexSlice[iso3, hazard, plot_rp, :, 'na', 'not_helped'], ['dw', 'dW_long_term']].mean().values

    plot_data[['dk', 'k']] /= plot_data.k.sum()
    plot_data[['y_bl', 'y', 'dy', 'tr_bl', 'tr', 'dtr', 'c_bl', 'c', 'dc_short_term', 'dc_savings', 'dc_reco', 'dc_income', 'dc_long_term', 'dc_income_sp', 'dc_income_lab', 'c_income_sp', 'c_income_lab', 'c_income']] /= plot_data.y_bl.sum()
    # plot_data[['c_bl', 'c', 'dc_short_term', 'dc_savings', 'dc_reco', 'dc_income', 'dc_long_term', 'dc_income_sp', 'dc_income_lab', 'c_income_sp', 'c_income_lab', 'c_income']] /= plot_data.c_bl.sum()
    plot_data[['dw_short_term', 'dw_long_term', 'dw']] /= plot_data.dw.sum()

    fig_width = double_col_width * centimeter
    fig_height = 0.5 * fig_width
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
        index=["a1", "a2", "a3", "a4", "a5", "y1", "y2", "y3", "y4", "y5", "c1", "c2", "c3", "c4", "c5", "w1_st", "w2_st", "w3_st", "w4_st", "w5_st", "w1_lt", "w2_lt", "w3_lt", "w4_lt", "w5_lt", "t", "a_avg", "y_avg", "c_avg", "w_avg_st", "w_avg_lt"]
    )
    nodes.loc[["a1", "a2", "a3", "a4", "a5", "a_avg"], 'x0'] = x_assets - node_width / 2
    nodes.loc[["y1", "y2", "y3", "y4", "y5", "y_avg"], 'x0'] = x_output - node_width / 2
    nodes.loc[["c1", "c2", "c3", "c4", "c5", "c_avg"], 'x0'] = x_consumption - node_width / 2
    nodes.loc[["w1_st", "w2_st", "w3_st", "w4_st", "w5_st", "w_avg_st"], 'x0'] = x_wellbeing_loss - node_width / 2
    nodes.loc[["w1_lt", "w2_lt", "w3_lt", "w4_lt", "w5_lt", "w_avg_lt"], 'x0'] = x_wellbeing_loss - node_width / 2
    nodes.loc["t", 'x0'] = x_transfers - node_width / 2
    nodes.loc[:, 'x1'] = nodes.loc[:, 'x0'] + node_width
    nodes.loc[:, 'label'] = nodes.index
    nodes.loc[:, 'cum_outflows'] = 0
    nodes.loc[:, 'cum_inflows'] = 0

        # set the node extents
    y1 = np.array([1, 1, 1, 1]).reshape(4)
    for q in [1, 2, 3, 4, 5, '_avg']:
        nodes.loc[[f"a{q}", f"y{q}", f'c{q}', f'w{q}_st'], 'y1'] = y1
        node_heights = plot_data.loc[f'q{q}', ['k', 'y_bl', 'c_bl', 'dw_short_term']].values
        nodes.loc[[f"a{q}", f"y{q}", f'c{q}', f'w{q}_st'], 'y0'] = y1 - node_heights
        # y1 = y1 - (node_heights + node_pad)
        y1 = y1 - (node_heights + np.array([0, 0, 0, plot_data.loc[f"q{q}", "dw_long_term"]]) + node_pad)
        nodes.loc[f"w{q}_lt", 'y1'] = nodes.loc[f"w{q}_st", 'y0']
        nodes.loc[f"w{q}_lt", 'y0'] = nodes.loc[f"w{q}_lt", 'y1'] - plot_data.loc[f"q{q}", "dw_long_term"]

    q_avg_transfers = plot_data.loc['q_avg', 'tr']

    transfer_break_diameter = 0.05
    nodes.loc["t", "y1"] = nodes.loc[['y5', 'c5', 'a5'], 'y0'].min() - 2 * node_pad
    nodes.loc["t", "y0"] = nodes.loc["t", "y1"] - plot_data.drop('q_avg')[['tr', 'dtr']].sum().sum() - q_avg_transfers - transfer_break_diameter

    nodes.loc[['a_avg', 'y_avg', 'c_avg', 'w_avg_st'], 'y1'] = nodes.loc["t", "y0"] - 2 * node_pad
    nodes.loc['a_avg', 'y0'] = nodes.loc['a_avg', 'y1'] - plot_data.loc['q_avg', 'k']
    nodes.loc['y_avg', 'y0'] = nodes.loc['y_avg', 'y1'] - plot_data.loc['q_avg', 'y']
    nodes.loc['c_avg', 'y0'] = nodes.loc['c_avg', 'y1'] - plot_data.loc['q_avg', 'c']
    nodes.loc['w_avg_st', 'y0'] = nodes.loc['w_avg_st', 'y1'] - plot_data.loc['q_avg', 'dw_short_term']
    nodes.loc[f"w_avg_lt", 'y1'] = nodes.loc[f"w_avg_st", 'y0']
    nodes.loc[f"w_avg_lt", 'y0'] = nodes.loc[f"w_avg_lt", 'y1'] - plot_data.loc[f"q_avg", "dw_long_term"]
    # nodes.loc[['y_avg', 'c_avg'], 'color'] = 'w'

    nodes.loc[:, 'color'] = 'k'
    nodes.loc[["w1_lt", "w2_lt", "w3_lt", "w4_lt", "w5_lt", "w_avg_lt"], 'color'] = 'grey'

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

    def add_link(source, target, left, right, color, alpha, kernel_size):
        left_y1 = nodes.loc[source, 'y1'] - nodes.loc[source, 'cum_outflows']
        left_y0 = left_y1 - left
        nodes.loc[source, 'cum_outflows'] += left
        right_y1 = nodes.loc[target, 'y1'] - nodes.loc[target, 'cum_inflows']
        right_y0 = right_y1 - right
        nodes.loc[target, 'cum_inflows'] += right
        links.loc[len(links)] = [source, target, left_y0, left_y1, right_y0, right_y1, color, alpha, kernel_size]

    # set the links
    for q in [1, 2, 3, 4, 5, '_avg']:
        q_data = plot_data.loc[f'q{q}']
        add_link(f"a{q}", f'y{q}', q_data.dk, q_data.dy, 'red', .25, 50)
        add_link(f"a{q}", f'y{q}', (q_data.k - q_data.dk), q_data.y, 'green', .25, 50)
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
        if q == 5:
            nodes.loc["t", ["cum_outflows", "cum_inflows"]] += transfer_break_diameter
    # add_link("y_avg", "t", q_avg_transfers, q_avg_transfers, 'green', 50)
    # add_link("t", "c_avg", q_avg_transfers * ntl_asset_losses_rel, q_avg_transfers * ntl_asset_losses_rel, 'red', 50)
    # add_link("t", "c_avg", q_avg_transfers * (1 - ntl_asset_losses_rel), q_avg_transfers * (1 - ntl_asset_losses_rel), 'green', 50)
    # add_link("y_avg", "c_avg", q_avg_output * (1 - plot_data.tau_tax.iloc[0]), q_avg_output * (1 - plot_data.tau_tax.iloc[0]), 'green', 50)

    node_boxes = [Rectangle((row.x0, row.y0), row.x1 - row.x0, row.y1 - row.y0) for idx, row in nodes.iterrows()]
    node_boxes = node_boxes + [Polygon(transfer_break_coords, closed=True)]
    # nodes_pc = PatchCollection(node_boxes, facecolors=nodes.color, edgecolors='none')
    nodes_pc = PatchCollection(node_boxes, facecolors=nodes.color.to_list() + ['w'], edgecolors='none')
    ax.add_collection(nodes_pc)

    # welfare_boxes = []
    # for q in [2, 3, 4, 5, '_avg']:
    #     coords = np.array(
    #         [[nodes.loc[f'c{q}', 'x1'], nodes.loc[f'c{q}', 'y0']],
    #          [nodes.loc[f'c{q}', 'x1'], nodes.loc[f'c{q}', 'y1']],
    #          [nodes.loc[f'w{q}', 'x0'], nodes.loc[f'w{q}', 'y1']],
    #          [nodes.loc[f'w{q}', 'x0'], nodes.loc[f'w{q}', 'y0']]]
    #     )
    #     welfare_boxes.append(Polygon(coords, closed=True))
    # welfare_boxes_pc = PatchCollection(welfare_boxes, facecolors='grey', edgecolors='none', alpha=.4)
    # ax.add_collection(welfare_boxes_pc)

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

    for label, x, y in zip(['Assets', 'Output', 'Transfers', 'Consumption', 'Wellbeing losses'], [x_assets, x_output, x_transfers, x_consumption, x_wellbeing_loss], [1.05, 1.05, 1.05, 1.05, 1.05]):
        ax.text(x, y, label, ha='center', va='center', fontsize=8, fontweight='bold')

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
    ).loc[pd.IndexSlice[iso3, hazard, plot_rp, 'q5', 'a', 'not_helped']]

    plot_recovery(3, inset_plot_data.avg_prod_k, inset_plot_data.tau_tax, inset_plot_data.k, inset_plot_data.dk,
                  inset_plot_data.lambda_h, inset_plot_data.reconstruction_share_sigma_h, inset_plot_data.liquidity,
                  0, np.nan, inset_plot_data.recovery_params, inset_plot_data.gamma_SP * inset_plot_data.n,
                  inset_plot_data.diversified_share, axs=[inset_ax], show_ylabel=True, plot_capital=False,
                  ylims=[(0, inset_plot_data.c * 1.05), None], plot_legend=False)

    for text, xy, xytext in zip(['Income loss', 'Reconstruction loss', 'Liquidity'], [(0.2, 0.85), (0.145, 0.6), (0.09, 0.15)], [(0.6, 0.55), (0.55, 0.4), (0.5, 0.25)]):
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

    # plt.tight_layout()

    plt.show(block=False)


