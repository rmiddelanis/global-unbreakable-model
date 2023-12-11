from functools import partial

import numpy as np
import pandas as pd
import os

# possible materials (level 1) for GEM taxonomy; from GEM Building Taxonomy Version 2.0
GEM_MATERIAL_NAMES = {'MAT99': 'Unknown material', 'C99': 'Concrete, unknown reinforcement',
                       'CU': 'Concrete, unreinforced', 'CR': 'Concrete, reinforced',
                       'SRC': 'Concrete, composite with steel section', 'S': 'Steel', 'ME': 'Metal (except steel)',
                       'M99': 'Masonry, unknown reinforcement', 'MUR': 'Masonry, unreinforced',
                       'MCF': 'Masonry, confined', 'MR': 'Masonry, reinforced', 'E99': 'Earth, unknown reinforcement',
                       'EU': 'Earth, unreinforced', 'ER': 'Earth, reinforced', 'W': 'Wood', 'MATO': 'Other material'}

GEM_MATERIALS_FIRST_LEVEL = ['C99', 'CU', 'CR', 'SRC', 'S', 'ME', 'M99', 'MUR', 'MCF', 'MR', 'E99', 'EU', 'ER', 'W',
                             'MATO']
GEM_MATERIALS_SECOND_LEVEL = ['MUN99', 'ADO', 'ST99', 'STRUB', 'STDRE', 'CL99', 'CLBRS', 'CLBRH', 'CLBLH', 'CB99',
                              'CBS', 'CBH', 'MO', 'MR99', 'RS', 'RW', 'RB', 'RCM', 'RBC', 'ET99', 'ETR', 'ETC', 'ETO',
                              'W99', 'WHE', 'WLI', 'WS', 'WWD', 'WBB', 'WO']
GEM_MATERIALS_THIRD_LEVEL = ['MO99', 'MON', 'MOM', 'MOL', 'MOC', 'MOCL', 'SP99', '']

HAZUS_COUNTRIES = ['VIR', 'PRI', 'CAN', 'USA']


def gather_gem_data(root_dir):
    # Initialize an empty DataFrame
    main_df = pd.DataFrame()

    vars_to_keep = {'ID_0': 'iso3', 'NAME_0': 'country', 'OCCUPANCY': 'building_type',
                    'MACRO_TAXO': 'macro_taxonomy', 'TAXONOMY': 'taxonomy', 'BUILDINGS': 'n_buildings',
                    'DWELLINGS': 'n_dwellings', 'OCCUPANTS_PER_ASSET': 'occupants_per_asset',
                    'TOTAL_AREA_SQM': 'total_area_sqm'}
    index_vars = ['ID_0', 'NAME_0', 'OCCUPANCY', 'MACRO_TAXO', 'TAXONOMY']

    # Walk through root_dir
    for dirpath, dirnames, filenames in os.walk(root_dir):
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

                # Append df to main_df
                main_df = pd.concat([main_df, df])
    main_df.rename(vars_to_keep, axis=1, inplace=True)
    # main_df.index.names = [vars_to_keep[n] for n in main_df.index.names]
    return main_df


def get_material_from_taxonomy(taxonomy, keep_subcategories=False):
    # HAZUS taxonomy
    if '-' in taxonomy and '/' not in taxonomy:
        return taxonomy

    # GEM taxonomy
    # see also GEM taxonomy tool at https://platform.openquake.org/taxtweb/
    elif '/' in taxonomy:
        attributes = taxonomy.split('/')
        if len(attributes) > 0:
            if 'DX' in attributes[0] and len(attributes) > 1:
                material_string = attributes[1]
            else:
                material_string = attributes[0]
            if not keep_subcategories:
                material_string = material_string.split('+')[0] if 'MIX' not in material_string else material_string
            if sum([m == material_string.split('+')[0] for m in GEM_MATERIAL_NAMES]) != 1:
                print(f"Warning: Unknown material {material_string} for taxonomy {taxonomy}.")
        else:
            print(f"Warning: no attributes found in taxonomy {taxonomy}.")
            material_string = ''
        return material_string


def compute_taxonomy_correspondences(gem_data):
    """
    This function computes the correspondence between macro taxonomy and derived material.

    Parameters:
    gem_data (DataFrame): The DataFrame containing the GEM data.

    Returns:
    DataFrame: A DataFrame containing the correspondence between macro taxonomy and derived material.
    """

    # Filter out the rows where 'iso3' is not in HAZUS_COUNTRIES and
    # keep only 'macro_taxonomy', 'taxonomy', 'n_buildings' columns
    matching = gem_data[~gem_data['iso3'].isin(HAZUS_COUNTRIES)][['macro_taxonomy', 'taxonomy', 'n_buildings']]

    # Apply the function 'get_material_from_taxonomy' to the 'taxonomy' column and store the result in a new column
    # 'derived_material'
    matching['derived_material'] = matching.taxonomy.apply(partial(get_material_from_taxonomy, keep_subcategories=True))

    # Create result DataFrame with unique 'macro_taxonomy' as index and unique 'derived_material' as columns
    res = pd.DataFrame(index=gem_data.macro_taxonomy.unique(), columns=matching.derived_material.unique())
    res.index.name = 'macro_taxonomy'
    res.columns.name = 'derived_material'

    # For each 'macro_taxonomy', compute the sum of 'n_buildings' grouped by 'derived_material' and normalize it
    for macro_taxo in res.index:
        n_buildings = matching[matching.macro_taxonomy == macro_taxo].groupby('derived_material').n_buildings.sum()
        n_buildings = n_buildings / n_buildings.sum()
        res.loc[macro_taxo, n_buildings.index] = n_buildings.values

    # shape the result DataFrame
    res = res.stack().reset_index().sort_values(by=['macro_taxonomy', 0], ascending=False).reset_index(drop=True)

    # Rename
    res.rename({0: 'share'}, axis=1, inplace=True)

    # Create a new column 'derived_material_l1' by splitting 'derived_material' on '+' and keeping the first level
    res['derived_material_l1'] = res.derived_material.apply(lambda s: s.split('+')[0] if '(' not in s else s)

    # Set 'macro_taxonomy', 'derived_material_l1', 'derived_material' as the index of the DataFrame
    res.set_index(['macro_taxonomy', 'derived_material_l1', 'derived_material'], inplace=True)

    return res


def get_gem_to_vulnerability_mapping():
    # PAGER category descriptions
    pager_desc = pd.read_csv("./pager_description.csv").astype(str)
    # PAGER to category mapping
    pager_category_mapping = pd.read_csv("./pager_description_to_aggregate_category.csv").astype(str)
    # PAGER id to category mapping
    pager_category_mapping = pd.merge(pager_desc, pager_category_mapping, on='pager_description')

    # PAGER to GEM
    pager_gem_mapping = pd.read_csv("./pager-gem_mapping.csv").astype(str)
    pager_gem_mapping['material'] = pager_gem_mapping.gem_str.apply(lambda s: s.split('/')[1].split('+')[0])

    # assign PAGER categories to GEM:
    res = pd.merge(pager_gem_mapping, pager_category_mapping, on='pager_id')[['material', 'aggregate_category']]
    res['count'] = 1
    res = res.groupby(['material', 'aggregate_category']).count()['count'].unstack().fillna(0)
    total = res.sum(axis=1)
    res_rel = res.copy()
    for c in res_rel.columns:
        res_rel[c] = res_rel[c] / total
    return res, res_rel