import os
import yaml

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_outpath = os.path.join(root_dir, "results/simulation_output")
gadm_path = os.path.join(root_dir, "data/raw/GADM/gadm_410-levels.gpkg")
figure_outpath = os.path.join(root_dir, "results/figures")
model_path = os.path.join(root_dir, "src/model/run_model.py")
force_recompute_on_first_simulation = True
force_download_on_first_simulation = False

def run_simulation(simulation_path_, settings_):
    if not os.path.exists(simulation_path_):
        os.makedirs(simulation_path_)
    global force_recompute_on_first_simulation, force_download_on_first_simulation
    settings_['scenario_params']['run_params']['force_recompute'] = force_recompute_on_first_simulation
    settings_['scenario_params']['run_params']['download'] = force_download_on_first_simulation
    with open(os.path.join(simulation_path_, 'settings.yml'), 'w') as f:
        yaml.dump(settings_, f)
    force_recompute_on_first_simulation = False
    force_download_on_first_simulation = False
    execute = f"python {model_path} {simulation_path_}/settings.yml"
    execute_res = os.system(execute)
    return execute_res


def default_settings():
    settings_ = {
        'model_params': {
            'run_params': {
                'append_date': False,
                'outpath': '',
                'verbose': False
            },
            'pds_params': {
                'pds_variant': 'no',
                'pds_targeting': 'perfect',
                'pds_borrowing_ability': 'unlimited',
                'covered_loss_share': 0.,
                'pds_lending_rate': 0.05,
                'pds_scope': [[0, .2], [.2, .4], [.4, .6], [.6, .8], [.8, 1]],
            }
        },
        'scenario_params': {
            'run_params': {
                'outpath': '',
                'force_recompute': False,
                'download': False,
                'verbose': False,
                'countries': 'all',
                'hazards': 'all',
                'num_cores': 8,
                'exclude_countries': ['THA'],
                'resolution': .2,
            },
            'macro_params': {
                'income_elasticity_eta': 1.5,
                'discount_rate_rho': 0.06,
                'axfin_impact': 0.1,
                'reduction_vul': 0.2
            },
            'hazard_params': {
                'hazard_protection': 'FLOPROS',
                'no_exposure_bias': False,
                'fa_threshold': 0.99
            },
            'policy_params': {
                'scale_self_employment': {
                    'parameter': 1,
                    'scope': [[0, .2], [.2, .4], [.4, .6], [.6, .8], [.8, 1]]
                },
                'scale_non_diversified_income': {
                    'parameter': 1,
                    'scope': [[0, .2], [.2, .4], [.4, .6], [.6, .8], [.8, 1]]
                },
                'min_diversified_share': {
                    'parameter': 0,
                    'scope': [[0, .2], [.2, .4], [.4, .6], [.6, .8], [.8, 1]]
                },
                'scale_income': {
                    'parameter': 1,
                    'scope': [[0, .2], [.2, .4], [.4, .6], [.6, .8], [.8, 1]]
                },
                'scale_liquidity': {
                    'parameter': 1,
                    'scope': [[0, .2], [.2, .4], [.4, .6], [.6, .8], [.8, 1]]
                },
                'scale_exposure': {
                    'parameter': 1,
                    'scope': [[0, .2], [.2, .4], [.4, .6], [.6, .8], [.8, 1]],
                    'scale_total': False
                },
                'scale_vulnerability': {
                    'parameter': 1,
                    'scope': [[0, .2], [.2, .4], [.4, .6], [.6, .8], [.8, 1]],
                    'scale_total': False
                }
            }
        }
    }
    return settings_


# Run baseline / PDS simulations
pds_params = {}
pds_params['0_baseline'] = {
    'pds_variant': 'no',
    'pds_targeting': 'perfect',
    'pds_borrowing_ability': 'unlimited',
    'pds_lending_rate': 0.05,
    'pds_scope': [[0, .2], [.2, .4], [.4, .6], [.6, .8], [.8, 1]],
    'covered_loss_share': [0.],
}
pds_params['8_post_disaster_support'] = {
    'pds_variant': 'unif_poor',
    'pds_targeting': 'perfect',
    'pds_borrowing_ability': 'unlimited',
    'pds_lending_rate': 0.05,
    'pds_scope': [[0, .2], [.2, .4], [.4, .6], [.6, .8], [.8, 1]],
    'covered_loss_share': [.4],
}
pds_params['9_insurance'] = {
    'pds_variant': 'proportional',
    'pds_targeting': 'perfect',
    'pds_borrowing_ability': 'unlimited',
    'pds_lending_rate': 0.05,
    'pds_scope': [[0, .2], [.2, .4], [.4, .6], [.6, .8], [.8, 1]],
    'covered_loss_share': [.2],
}

for pds_name, pds_dict in pds_params.items():
    for covered_loss_share in pds_dict['covered_loss_share']:
        settings = default_settings()
        pds_dict_ = pds_dict.copy()
        pds_dict_['covered_loss_share'] = covered_loss_share
        if pds_name != '0_baseline':
            simulation_path = os.path.join(model_outpath, pds_name, '0-1', str(covered_loss_share))
        else:
            simulation_path = os.path.join(model_outpath, pds_name)
        settings['model_params']['run_params']['outpath'] = simulation_path
        settings['scenario_params']['run_params']['outpath'] = simulation_path
        settings['model_params']['pds_params'] = pds_dict_
        print(f"Starting simulation '{pds_name}' with shareable '{covered_loss_share}'")
        run_simulation(simulation_path, settings)


# Run policy simulations
policy_params = {
    '1_reduce_total_exposure': {
        'measure': ['scale_exposure'],
        'parameters': [0.95],
        'scopes': [[[0, .2]], [[0, .2], [.2, .4], [.4, .6], [.6, .8], [.8, 1]]],
        'scale_total': True,
    },
    '2_reduce_total_vulnerability': {
        'measure': ['scale_vulnerability'],
        'parameters': [.95],
        'scopes': [[[0, .2]], [[0, .2], [.2, .4], [.4, .6], [.6, .8], [.8, 1]]],
        'scale_total': True,
    },
    '3_scale_income_and_liquidity': {
        'measure': ['scale_income', 'scale_liquidity'],
        'parameters': [1.05],
        'scopes': [[[0, .2], [.2, .4], [.4, .6], [.6, .8], [.8, 1]]],
    },
    '4_scale_self_employment': {
        'measure': ['scale_self_employment'],
        'parameters': [.9],
        'scopes': [[[0, .2], [.2, .4], [.4, .6], [.6, .8], [.8, 1]]],
    },
    '5_scale_non_diversified_income': {
        'measure': ['scale_non_diversified_income'],
        'parameters': [.9],
        'scopes': [[[0, .2], [.2, .4], [.4, .6], [.6, .8], [.8, 1]]],
    },
    '6_scale_liquidity': {
        'measure': ['scale_liquidity'],
        'parameters': [0],
        'scopes': [[[0, .2], [.2, .4], [.4, .6], [.6, .8], [.8, 1]]],
    },
    '7_scale_gini_index': {
        'measure': ['scale_gini_index'],
        'parameters': [.9],
        'scopes': [None],
    },
}

for policy, policy_dict in policy_params.items():
    for scope in policy_dict['scopes']:
        for parameter in policy_dict['parameters']:
            settings = default_settings()
            if scope is not None:
                scope_folder = '__'.join([f"{s[0]}-{s[1]}" for s in scope])
            else:
                scope_folder = '0-1'
            simulation_path = os.path.join(model_outpath, policy, scope_folder, str(parameter))
            settings['model_params']['run_params']['outpath'] = simulation_path
            settings['scenario_params']['run_params']['outpath'] = simulation_path
            for policy_element in policy_dict['measure']:
                # in each simulation, apply all 'measures', but only one 'parameter' and 'scope'
                settings['scenario_params']['policy_params'][policy_element] = {
                    'parameter': parameter,
                }
                if scope is not None:
                    settings['scenario_params']['policy_params'][policy_element]['scope'] = scope
                if 'scale_total' in policy_dict:
                    settings['scenario_params']['policy_params'][policy_element]['scale_total'] = True
            print(f"Starting simulation with policy '{policy}', parameter '{parameter}', and scope '{scope}'")
            run_simulation(simulation_path, settings)

# generate figures
if os.path.exists(gadm_path):
    os.system(f"python src/misc/plotting.py {model_outpath} {figure_outpath} --gadm_path {gadm_path} --plot")
else:
    print("GADM data not found.")
    os.system(f"python src/misc/plotting.py {model_outpath} {figure_outpath} --plot")