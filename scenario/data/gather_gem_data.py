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


import json
import os
import tqdm
import pandas as pd
import numpy as np

HAZUS_COUNTRIES = ['VIR', 'PRI', 'CAN', 'USA']


def load_mapping(gem_fields_path_, vuln_class_mapping_):
    """
    Loads the GEM (GLobal Exposure Model) taxonomy fields and vulnerability class mapping.

    Args:
        gem_fields_path_ (str): Path to the GEM taxonomy fields JSON file.
        vuln_class_mapping_ (str): Path to the vulnerability class mapping Excel file.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: DataFrame with the vulnerability class mapping.
            - dict: Dictionary mapping field values to their types.
    """
    gem_fields = json.load(open(gem_fields_path_, 'r'))
    field_value_to_type_map = {v: k.lower() for k in gem_fields.keys() for l in gem_fields[k].keys() for v in
                               gem_fields[k][l]}

    mapping_df = pd.read_excel(vuln_class_mapping_, header=0)
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
    return mapping_df, field_value_to_type_map


def assign_vulnerability(material, resistance_system, height, mapping, verbose=True):
    """
    Assigns a vulnerability class based on material, resistance system, and height.

    Args:
        material (str): Material type.
        resistance_system (str): Lateral load resistance system.
        height (str): Building height.
        mapping (pd.DataFrame): DataFrame containing the vulnerability class mapping.
        verbose (bool): Whether to print warnings. Defaults to True.

    Returns:
        pd.Series: A Series containing the assigned vulnerability class.

    Raises:
        ValueError: If the material is unknown or cannot be mapped.
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
                            if verbose:
                                print(f"Warning: could not parse height value {height} to integer. Using default value.")
                return mapping.loc[(material, resistance_system, 'default')].transpose().squeeze().rename('vulnerability')
            return mapping.loc[(material, 'default')].transpose().squeeze().rename('vulnerability')
    else:
        raise ValueError(f"Could not assign vulnerability for unknown material {material}.")


def decode_taxonomy(taxonomy, field_value_to_type_map, keep_unknown=False, verbose=True):
    """
    Decodes a GEM taxonomy string into its components.

    Args:
        taxonomy (str): GEM taxonomy string.
        field_value_to_type_map (dict): Mapping of field values to their types.
        keep_unknown (bool): Whether to keep unknown attributes. Defaults to False.
        verbose (bool): Whether to print warnings. Defaults to True.

    Returns:
        pd.DataFrame: DataFrame with decoded taxonomy components.
    """
    res = pd.DataFrame({col: [[]] for col in ['lat_load_mat', 'lat_load_sys', 'height', 'unknown']},
                       index=[taxonomy])
    res.index.name = 'taxonomy'
    attribute_types = {attribute: identify_gem_attribute_type(attribute, field_value_to_type_map, verbose)
                       for attribute in taxonomy.split('/')}
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
    if keep_unknown:
        return res
    else:
        return res.drop('unknown', axis=1)


def identify_gem_attribute_type(attribute, field_value_to_type_map, verbose=True):
    """
    Identifies the type of a GEM attribute.

    Args:
        attribute (str): GEM attribute string.
        field_value_to_type_map (dict): Mapping of field values to their types.
        verbose (bool): Whether to print warnings. Defaults to True.

    Returns:
        np.ndarray: Array of identified attribute types.
    """
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



def gather_gem_data(gem_repo_root_dir_, hazus_gem_mapping_path_, gem_fields_path_, vuln_class_mapping_,
                    vulnerability_class_output_=None, weight_by='total_replacement_cost', verbose=True):
    """
    Gathers and processes GEM (Global Exposure Model) data.

    Args:
        gem_repo_root_dir_ (str): Root directory of the GEM repository.
        hazus_gem_mapping_path_ (str): Path to the HAZUS-GEM mapping CSV file.
        gem_fields_path_ (str): Path to the GEM taxonomy fields JSON file.
        vuln_class_mapping_ (str): Path to the vulnerability class mapping Excel file.
        vulnerability_class_output_ (str, optional): Path to save the vulnerability class distribution. Defaults to None.
        weight_by (str): Column to use for weighting vulnerability distribution. Defaults to 'total_replacement_cost'.
        verbose (bool): Whether to print warnings. Defaults to True.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: GEM data with decoded taxonomy and assigned vulnerabilities.
            - pd.DataFrame: Vulnerability class shares by country and hazard.
    """

    # Initialize an empty DataFrame
    gem = pd.DataFrame()

    vars_to_keep = {
        'ID_0': 'iso3', 'NAME_0': 'country', 'OCCUPANCY': 'building_type', 'MACRO_TAXO': 'macro_taxonomy',
        'TAXONOMY': 'taxonomy', 'BUILDINGS': 'n_buildings',  # 'DWELLINGS': 'n_dwellings',
        # 'OCCUPANTS_PER_ASSET': 'occupants_per_asset',
        'TOTAL_AREA_SQM': 'total_area_sqm', 'TOTAL_REPL_COST_USD': 'total_replacement_cost',
        'COST_CONTENTS_USD': 'contents_cost', 'COST_STRUCTURAL_USD': 'structural_cost',
        'COST_NONSTRUCTURAL_USD': 'nonstructural_cost',

    }
    index_vars = ['ID_0', 'NAME_0', 'OCCUPANCY', 'MACRO_TAXO', 'TAXONOMY']

    # Walk through root_dir
    for dirpath, dirnames, filenames in os.walk(gem_repo_root_dir_):
        for filename in filenames:
            # Check if the file is 'Exposure_Summary_Taxonomy.csv'
            if filename == 'Exposure_Summary_Taxonomy.csv':
                # Construct the full file path
                file_path = str(os.path.join(dirpath, filename))

                # Read the file into a DataFrame
                df = pd.read_csv(file_path)

                vars_diff = np.setdiff1d(list(vars_to_keep.keys()), df.columns)
                if len(vars_diff) > 0:
                    print(f"Warning: the following variables aren't available for country {os.path.basename(dirpath)}: "
                          + ", ".join(list(vars_diff)))
                df = df[list(set(vars_to_keep.keys()) - set(vars_diff))].groupby(index_vars).sum().reset_index()

                # Append df to gem_data
                gem = pd.concat([gem, df])
    gem.rename(vars_to_keep, axis=1, inplace=True)
    gem.reset_index(inplace=True, drop=True)

    replace_strings = {s: s.replace('+', '-') for s in
                       ['MIX(MUR+W)', 'MIX(MR+W)', 'MIX(S+CR)', 'MIX(MUR+CR)', 'MIX(W+EU)', 'MIX(MUR+STRUB+W)',
                        'MIX(MUR+STDRE+W)', 'MIX(S+CR+PC)']}
    for s, r in replace_strings.items():
        gem.taxonomy = gem.taxonomy.apply(lambda x: x.replace(s, r))

    # handle countries with HAZUS taxonomy:
    # countries that use HAZUS have a taxonomy of the form "{occupancy}-{HAZUS id/[height]}-{Design Code}"
    # use hazus id to replace taxonomy with respective GEM taxonomy string
    # some entries come with an additional height value. This is mostly consistent with the GEM taxonomy strings as
    # per the GEM Building Taxonomy Version 2.0 table D-2, except for some W1 Hazus IDs, which have height > 2 stories
    # information on Design Codes can be found in the following document (page 2-4, section 2.3):
    # https://www.fema.gov/sites/default/files/2020-09/fema_hazus_advanced-engineering-building-module_user-manual.pdf
    hazus_gem_mapping = pd.read_csv(hazus_gem_mapping_path_, index_col=0).astype(str)
    # set mobile homes to informal
    hazus_gem_mapping.loc['MH', 'gem_str'] = 'INF/'
    # W3 and W4 are not allowed as per Hazus documentation, but occur in the dataset; setting to general 'Wood'
    hazus_gem_mapping.loc['W3', 'gem_str'] = 'W/'
    hazus_gem_mapping.loc['W4', 'gem_str'] = 'W/'
    gem.loc[gem.iso3.isin(HAZUS_COUNTRIES), 'taxonomy'] = (
        gem.loc[gem.iso3.isin(HAZUS_COUNTRIES), 'taxonomy'].apply(
            lambda x: hazus_gem_mapping.loc[x.split('-')[1].split('/')[0], 'gem_str']
        )
    )

    vulnerability_mapping, field_value_to_type_map = load_mapping(gem_fields_path_=gem_fields_path_,
                                                                  vuln_class_mapping_=vuln_class_mapping_)

    unique_tax_strings = gem.taxonomy.unique()
    decoded_tax_strings = pd.concat(
        [decode_taxonomy(t, field_value_to_type_map, keep_unknown=False, verbose=verbose)
         for t in tqdm.tqdm(unique_tax_strings, desc="decoding taxonomy strings")]
    )
    res = pd.merge(gem, decoded_tax_strings, how='left', on='taxonomy')

    # set material to 'UNK' if Lateral load resisting system value = 'LN' (No lateral load-resisting system)
    res.loc[(res.lat_load_mat.isna())
                     & (res.lat_load_sys.apply(lambda x: 'LN' in x if type(x) is str else False)), "lat_load_mat"] = 'UNK'
    # if taxonomy starts with 'UNK', assume this is the material code and set material to 'UNK'
    res.loc[(res.lat_load_mat.isna()) & (res.taxonomy.apply(lambda x: x.startswith('UNK'))), "lat_load_mat"] = 'UNK'

    # assign vulnerability classes
    vulnerability = res.apply(
        lambda x: assign_vulnerability(x.lat_load_mat, x.lat_load_sys, x.height, vulnerability_mapping, verbose=verbose), axis=1
    )
    merged = pd.concat([res, vulnerability], axis=1)

    v_class_shares = []
    for hazard_class in vulnerability.columns:
        v_class_shares_ = merged.groupby(['iso3', 'country', f'{hazard_class}'])[weight_by].sum()
        v_class_shares_ = v_class_shares_ / merged.groupby('iso3')[weight_by].sum()
        v_class_shares_ = v_class_shares_.unstack()
        v_class_shares_.fillna(0, inplace=True)
        v_class_shares_.columns = pd.MultiIndex.from_product([[hazard_class], v_class_shares_.columns])
        v_class_shares.append(v_class_shares_)
    v_class_shares = pd.concat(v_class_shares, axis=1)
    if vulnerability_class_output_:
        v_class_shares.to_csv(vulnerability_class_output_)
    return merged, v_class_shares


if __name__ == '__main__':
    gem_repo_root_dir = './data/raw/GEM_vulnerability/global_exposure_model/'
    vulnarebility_class_mapping = "./data/raw/GEM_vulnerability/gem-to-vulnerability_mapping_per_hazard.xlsx"
    hazus_gem_mapping_path = './data/raw/GEM_vulnerability/hazus-gem_mapping.csv'
    gem_fields_path = "./data/raw/GEM_vulnerability/gem_taxonomy_fields.json"
    gem_data, vuln_class_shares = gather_gem_data(
        gem_repo_root_dir_=gem_repo_root_dir,
        hazus_gem_mapping_path_=hazus_gem_mapping_path,
        gem_fields_path_=gem_fields_path,
        vuln_class_mapping_=vulnarebility_class_mapping,
        vulnerability_class_output_=None,
        weight_by='total_replacement_cost',
        verbose=True,
    )
    print(gem_data)
    print(vuln_class_shares)
