import json
import os
import tqdm
import pandas as pd
import numpy as np

HAZUS_COUNTRIES = ['VIR', 'PRI', 'CAN', 'USA']


def load_mapping(gem_fields_path='inputs/GEM_vulnerability/gem_taxonomy_fields.json'):
    gem_fields = json.load(open(gem_fields_path, 'r'))
    field_value_to_type_map = {v: k.lower() for k in gem_fields.keys() for l in gem_fields[k].keys() for v in
                               gem_fields[k][l]}

    mapping_df = pd.read_excel("./inputs/GEM_vulnerability/gem-to-vulnerability_mapping_per_hazard.xlsx", header=0)
    mapping_df.drop('comment', inplace=True, axis=1)
    mapping_df.rename(columns={'combined': 'default'}, inplace=True)
    mapping_df.set_index(['lat_load_mat', 'lat_load_sys', 'height'], inplace=True)
    for c in mapping_df.columns:
        if '+' in c:
            for c_ in c.split('+'):
                if len(c_) > 1:
                    mapping_df[c_] = mapping_df[c]
            mapping_df = mapping_df.drop(c, axis=1)
    mapping_df = mapping_df.sort_index()
    # vulnerability_mapping = {}
    # for i, row in mapping_df.iterrows():
    #     row_mapping = row
    #     # row_mapping = row['hazard'].to_dict()['combined']
    #     lat_load_mat, lat_load_sys, height = i
    #     if lat_load_sys is np.nan:
    #         vulnerability_mapping[lat_load_mat] = row_mapping
    #     elif height is np.nan:
    #         if lat_load_mat not in vulnerability_mapping:
    #             vulnerability_mapping[lat_load_mat] = {lat_load_sys: row_mapping}
    #         else:
    #             vulnerability_mapping[lat_load_mat][lat_load_sys] = row_mapping
    #     else:
    #         if lat_load_mat not in vulnerability_mapping:
    #             vulnerability_mapping[lat_load_mat] = {lat_load_sys: {height: row_mapping}}
    #         elif lat_load_sys not in vulnerability_mapping[lat_load_mat]:
    #             vulnerability_mapping[lat_load_mat][lat_load_sys] = {height: row_mapping}
    #         else:
    #             vulnerability_mapping[lat_load_mat][lat_load_sys][height] = row_mapping

    # return vulnerability_mapping, field_value_to_type_map
    return mapping_df, field_value_to_type_map


def assign_vulnerability(material, resistance_system, height, mapping):#, new_approach=True):
    """
    This function assigns a vulnerability to a given GEM taxonomy.

    Parameters:
    gem_taxonomy (str): The GEM taxonomy.
    mapping (dict): The mapping between GEM taxonomy and vulnerability.

    Returns:
    str: The vulnerability.
    """
    # if new_approach:
    if material in mapping.index:
        if len(mapping.loc[material]) == 1:
            return mapping.loc[[material]].transpose().squeeze().rename('vulnerability')
        else:
            if type(resistance_system) is str and len(resistance_system) > 0:
                resistance_system = resistance_system.split('+')[0]
            if (material, resistance_system) in mapping.index:
                if len(mapping.loc[(material, resistance_system)]) == 1:
                    return mapping.loc[material, resistance_system].transpose().squeeze().rename('vulnerability')
                else:
                    if type(height) is str and len(height) > 0:
                        height = height.split(':')[1].split('+')[0].split('-' if '-' in height else ',')
                        if len(height) > 1 and len(height[1]) == 0 or len(height) == 1:
                            height = [height[0], height[0]]
                        try:
                            height = [int(h) for h in height]
                            for h_idx in mapping.loc[(material, resistance_system)].index:
                                if h_idx != 'default':
                                    h_range = sorted([int(h) for h in h_idx.split(':')[1].split(',')])
                                    if h_range[0] <= height[0] <= h_range[1] or h_range[0] <= height[1] <= h_range[1]:
                                        return mapping.loc[(material, resistance_system, h_idx)].transpose().squeeze().rename('vulnerability')
                        except ValueError as e:
                            print(f"Warning: could not parse height value {height} to integer. Using default value.")
                return mapping.loc[(material, resistance_system, 'default')].transpose().squeeze().rename('vulnerability')
            return mapping.loc[(material, 'default')].transpose().squeeze().rename('vulnerability')
    else:
        raise ValueError(f"Could not assign vulnerability for unknown material {material}.")
        # print(f"Could not assign vulnerability for unknown material {material}.")
        # return 'unknown'
    #
    # else:
    #     if material in mapping.keys():
    #         material_vulnerability = mapping[material]
    #         if type(material_vulnerability) is str:
    #             return material_vulnerability
    #         if type(resistance_system) is str and len(resistance_system) > 0:
    #             resistance_system = resistance_system.split('+')[0]
    #             if resistance_system in material_vulnerability.keys():
    #                 resistance_system_vulnerability = material_vulnerability[resistance_system]
    #                 if type(resistance_system_vulnerability) is str:
    #                     return resistance_system_vulnerability
    #                 else:
    #                     if type(height) is str and len(height) > 0:
    #                         height = height.split(':')[1].split('+')[0].split('-' if '-' in height else ',')
    #                         if len(height) > 1 and len(height[1]) == 0 or len(height) == 1:
    #                             height = [height[0], height[0]]
    #                         try:
    #                             height = [int(h) for h in height]
    #                         except ValueError as e:
    #                             print(f"Warning: could not parse height value {height} to integer. Using default value.")
    #                             return resistance_system_vulnerability['default']
    #                         for key in resistance_system_vulnerability.keys():
    #                             if key != 'default':
    #                                 h_range = sorted([int(h) for h in key.split(':')[1].split(',')])
    #                                 if h_range[0] <= height[0] <= h_range[1] or h_range[0] <= height[1] <= h_range[1]:
    #                                     return resistance_system_vulnerability[key]
    #                 return resistance_system_vulnerability['default']
    #         return material_vulnerability['default']
    #     # raise ValueError(f"Unknown material {material} for taxonomy {gem_taxonomy}.")
    #     print(f"Could not assign vulnerability for unknown material {material}.")
    #     return 'unknown'


def gather_gem_data(gem_repo_root_dir, hazus_gem_mapping_path, gem_fields_path, vulnerability_class_output=None,
                    weight_by='replacement_cost'):
    """
        This function gathers GEM (Global Exposure Model) data from the GEM repository directory, decodes the taxonomy
        strings, assigns vulnerabilities based on the decoded taxonomy, and optionally outputs the distribution of
        vulnerabilities per country.

        Parameters:
        gem_repo_root_dir (str): The root directory of the GEM repository.
        hazus_gem_mapping_path (str): The path to the CSV file containing the mapping between HAZUS and GEM taxonomies.
        vulnerability_class_output (str, optional): The path to the output CSV file containing the distribution of
                                                    vulnerabilities per country. If None, no output is generated.
        weight_by (str, optional): The column to use for weighting when computing the distribution of vulnerabilities.
                                   Default is 'replacement_cost'.

        Returns:
        pandas.DataFrame: A DataFrame containing the GEM data with decoded taxonomy and assigned vulnerabilities.
        """

    # Initialize an empty DataFrame
    gem_data = pd.DataFrame()

    vars_to_keep = {
        'ID_0': 'iso3', 'NAME_0': 'country', 'OCCUPANCY': 'building_type', 'MACRO_TAXO': 'macro_taxonomy',
        'TAXONOMY': 'taxonomy', 'BUILDINGS': 'n_buildings',  # 'DWELLINGS': 'n_dwellings',
        # 'OCCUPANTS_PER_ASSET': 'occupants_per_asset', 'TOTAL_AREA_SQM': 'total_area_sqm',
        'TOTAL_REPL_COST_USD': 'replacement_cost',

    }
    index_vars = ['ID_0', 'NAME_0', 'OCCUPANCY', 'MACRO_TAXO', 'TAXONOMY']

    # Walk through root_dir
    for dirpath, dirnames, filenames in os.walk(gem_repo_root_dir):
        for filename in filenames:
            # Check if the file is 'Exposure_Summary_Taxonomy.csv'
            if filename == 'Exposure_Summary_Taxonomy.csv':
                # Construct the full file path
                file_path = os.path.join(dirpath, filename)

                # Read the file into a DataFrame
                df = pd.read_csv(file_path)

                vars_diff = np.setdiff1d(list(vars_to_keep.keys()), df.columns)
                if len(vars_diff) > 0:
                    print(f"Warning: the following variables aren't available for country {os.path.basename(dirpath)}: "
                          + ", ".join(list(vars_diff)))
                df = df[list(set(vars_to_keep.keys()) - set(vars_diff))].groupby(index_vars).sum().reset_index()

                # Append df to gem_data
                gem_data = pd.concat([gem_data, df])
    gem_data.rename(vars_to_keep, axis=1, inplace=True)
    gem_data.reset_index(inplace=True, drop=True)

    replace_strings = {s: s.replace('+', '-') for s in
                       ['MIX(MUR+W)', 'MIX(MR+W)', 'MIX(S+CR)', 'MIX(MUR+CR)', 'MIX(W+EU)', 'MIX(MUR+STRUB+W)',
                        'MIX(MUR+STDRE+W)', 'MIX(S+CR+PC)']}
    for s, r in replace_strings.items():
        gem_data.taxonomy = gem_data.taxonomy.apply(lambda x: x.replace(s, r))

    # handle countries with HAZUS taxonomy:
    # countries that use HAZUS have a taxonomy of the form "{occupancy}-{HAZUS id/[height]}-{Design Code}"
    # use hazus id to replace taxonomy with respective GEM taxonomy string
    # some entries come with an additional height value. This is mostly consistent with the GEM taxonomy strings as
    # per the GEM Building Taxonomy Version 2.0 table D-2, except for some W1 Hazus IDs, which have height > 2 stories
    # information on Design Codes can be found in the following document (page 2-4, section 2.3):
    # https://www.fema.gov/sites/default/files/2020-09/fema_hazus_advanced-engineering-building-module_user-manual.pdf
    hazus_gem_mapping = pd.read_csv(hazus_gem_mapping_path, index_col=0).astype(str)
    # set mobile homes to informal, s.th. vulnerability will be fragile
    hazus_gem_mapping.loc['MH', 'gem_str'] = 'INF/'
    # W3 and W4 are not allowed as per Hazus documentation, but occur in the dataset; setting to general 'Wood'
    hazus_gem_mapping.loc['W3', 'gem_str'] = 'W/'
    hazus_gem_mapping.loc['W4', 'gem_str'] = 'W/'
    gem_data.loc[gem_data.iso3.isin(HAZUS_COUNTRIES), 'taxonomy'] = (
        gem_data.loc[gem_data.iso3.isin(HAZUS_COUNTRIES), 'taxonomy'].apply(
            lambda x: hazus_gem_mapping.loc[x.split('-')[1].split('/')[0], 'gem_str']
        )
    )

    vulnerability_mapping, field_value_to_type_map = load_mapping(gem_fields_path=gem_fields_path)

    unique_tax_strings = gem_data.taxonomy.unique()
    decoded_tax_strings = pd.concat(
        [decode_taxonomy(t, field_value_to_type_map, keep_unknown=False, verbose=False)
         for t in tqdm.tqdm(unique_tax_strings, desc="decoding taxonomy strings")]
    )
    res = pd.merge(gem_data, decoded_tax_strings, how='left', on='taxonomy')

    # set material to 'UNK' if Lateral load resisting system value = 'LN' (No lateral load-resisting system)
    res.lat_load_mat[(res.lat_load_mat.isna()) & (res.lat_load_sys.apply(lambda x: 'LN' in x if type(x) is str else False))] = 'UNK'
    # if taxonomy starts with 'UNK', assume this is the material code and set material to 'UNK'
    res.lat_load_mat[(res.lat_load_mat.isna()) & (res.taxonomy.apply(lambda x: x.startswith('UNK')))] = 'UNK'

    # assign vulnerability classes
    vulnerability = res.apply(
        lambda x: assign_vulnerability(x.lat_load_mat, x.lat_load_sys, x.height, vulnerability_mapping), axis=1
    )
    merged = pd.concat([res, vulnerability], axis=1)

    vuln_class_shares = []
    for hazard_class in vulnerability.columns:
        vuln_class_shares_ = merged.groupby(['iso3', 'country', f'{hazard_class}'])[weight_by].sum()
        vuln_class_shares_ = vuln_class_shares_ / merged.groupby('iso3')[weight_by].sum()
        vuln_class_shares_ = vuln_class_shares_.unstack()
        vuln_class_shares_.fillna(0, inplace=True)
        vuln_class_shares_.columns = pd.MultiIndex.from_product([[hazard_class], vuln_class_shares_.columns])
        vuln_class_shares.append(vuln_class_shares_)
    vuln_class_shares = pd.concat(vuln_class_shares, axis=1)
    # res['vulnerability_class'] = res.apply(lambda x: assign_vulnerability(x.lat_load_mat, x.lat_load_sys, x.height,
    #                                                                       VULNERABILITY_MAPPING), axis=1)
    #
    # vuln_class_shares = res.groupby(['iso3', 'country', 'vulnerability_class'])[weight_by].sum()
    # vuln_class_shares = vuln_class_shares / res.groupby('iso3')[weight_by].sum()
    # vuln_class_shares = vuln_class_shares.unstack()
    # vuln_class_shares.fillna(0, inplace=True)
    if vulnerability_class_output:
        vuln_class_shares.to_csv(vulnerability_class_output)
    return res, vuln_class_shares


def decode_taxonomy(taxonomy, field_value_to_type_map, keep_unknown=False, verbose=True):
    res = pd.DataFrame({col: [[]] for col in ['lat_load_mat', 'lat_load_sys', 'height', 'unknown']},
                       index=[taxonomy])
    res.index.name = 'taxonomy'
    # # HAZUS taxonomy
    # if '-' in taxonomy and taxonomy[:3] in ['COM', 'RES', 'IND']:
    #     res.loc[taxonomy, 'hazus_id'] = [taxonomy.split('/')[0].split('-')[1]]
    # # GEM taxonomy
    # elif '/' in taxonomy:
    attribute_types = {attribute: identify_gem_attribute_type(attribute, field_value_to_type_map, verbose) for attribute in taxonomy.split('/')}
    for attribute, attribute_type in attribute_types.items():
        res.loc[[taxonomy], attribute_type] = (
                res.loc[taxonomy, attribute_type] +
                pd.DataFrame(index=[taxonomy], columns=attribute_type, data=[[[attribute]] * len(attribute_type)])
        )
    for col in res.columns:
        if len(res.loc[taxonomy, col]) == 0:
            res.loc[taxonomy, col] = np.nan
        elif len(res.loc[taxonomy, col]) == 1:
            res.loc[taxonomy, col] = res.loc[taxonomy, col][0]
        elif len(res.loc[taxonomy, col]) > 1:
            if res.loc[taxonomy, col][0] in ['MATO', 'UNK'] and 'UNK' in res.loc[taxonomy, col][0]:
                res.loc[taxonomy, col] = res.loc[taxonomy, col][0]
            elif verbose:
                print(f"Warning: Multiple attributes have been mapped to the same type for taxonomy {taxonomy}.")
    material, resistance_system, height = res.loc[taxonomy, ['lat_load_mat', 'lat_load_sys', 'height']]
    if keep_unknown:
        return res
    else:
        return res.drop('unknown', axis=1)


def identify_gem_attribute_type(attribute, field_value_to_type_map, verbose=True):
    if len(attribute) == 0 and verbose:
        print("Warning: Empty attribute.")
    types = np.unique([field_value_to_type_map.get(field.split(':')[0], 'unknown') for field in attribute.split('+')])
    if len(types) == 1 and 'unknown' in types and verbose:
        print(f"Warning: Unknown type for attribute {attribute}.")
    elif len(types) == 2 and 'unknown' in types:
        types = types[types != 'unknown']
    elif len(types) > 1 and verbose:
        print(f"Warning: Multiple types {types} for attribute {attribute}.")
    return types


if __name__ == '__main__':
    gem_repo_root_dir = '../global_exposure_model/'
    hazus_gem_mapping_path = './inputs/GEM_vulnerability/hazus-gem_mapping.csv'
    gem_fields_path = "./inputs/GEM_vulnerability/gem_taxonomy_fields.json"
    vulnerability_class_output = './inputs/GEM_vulnerability/country_vulnerability_classes.csv'
    gem_data, vuln_class_shares = gather_gem_data(
        gem_repo_root_dir=gem_repo_root_dir,
        hazus_gem_mapping_path=hazus_gem_mapping_path,
        gem_fields_path=gem_fields_path,
        vulnerability_class_output=vulnerability_class_output,
        weight_by='replacement_cost',
    )
    print(gem_data)
    print(vuln_class_shares)
