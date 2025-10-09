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


import os
import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray as rxr
import tqdm
import xarray as xr
import zipfile
from io import BytesIO


def process_flopros_data(flopros_path, population_path, shapefiles_path, outpath=None):#, wb_shapes_path, chn_shape_path, twn_shape_path):
    """
    Processes FLOPROS datay to compute national-level flood protection levels for riverine and coastal areas.

    Args:
        flopros_path (str): Path to the FLOPROS shapefile containing protection data.
        flopros_update_path (str): Path to the Excel file with updated modeled coastal protection data.
        flopros_update_shapes_path (str): Path to the shapefile containing geometries for the updated modeled data.
        population_path (str): Path to the population raster file for the year 2020.
        shapefiles_path (str): Path to the GADM geopackage file containing country boundaries.
        outpath (str, optional): Directory to save the processed protection levels as shapefile and CSV. Defaults to None.

    Returns:
        geopandas.GeoDataFrame: A GeoDataFrame containing national-level flood protection levels for riverine and coastal areas.

    Notes:
        - The function merges design and policy layers for FLOPROS data and incorporates updated modeled coastal protection levels.
        - Population-weighted protection levels are calculated and aggregated to the country level.
        - Outputs are saved to the specified directory if `outpath` is provided.
    """
    # Load flopros data
    flopros = gpd.read_file(os.path.join(flopros_path, "Scussolini_et_al_FLOPROS_shp_V1"))
    flopros = flopros[flopros.OBJECTID != 0]

    # Original data has no merged layer due to lack of the modeled layer. Merge available design and policy layers
    # need to replace 0 with nan first, otherwise inland regions would decrease a country's ntl. coastal protection
    flopros[['DL_Max_Co', 'DL_Min_Co', 'PL_Max_Co', 'PL_Min_Co']] = (
        flopros[['DL_Max_Co', 'DL_Min_Co', 'PL_Max_Co', 'PL_Min_Co']].replace({0: np.nan}))
    flopros['DL_PL_Co'] = flopros[['DL_Max_Co', 'DL_Min_Co']].mean(axis=1).fillna(flopros[['PL_Max_Co', 'PL_Min_Co']].mean(axis=1))

    # Load flopros modeled layer data for coastal protection
    flopros_mod_co = pd.read_excel(os.path.join(flopros_path, "Tiggeloven_et_al_2020_data/Results_adaptation_objectives/Protection_constant_Adaptation_objective_RCP4P5_SSP1.xlsx"), index_col=0)
    flopros_update_shapes = gpd.read_file(os.path.join(flopros_path, "Tiggeloven_et_al_2020_data/Results_adaptation_objectives/Countries_States_simplified.shp"))
    flopros_update_shapes.set_crs(flopros.crs, inplace=True)
    flopros_mod_co = pd.merge(flopros_update_shapes[['FID_Aque', 'geometry']], flopros_mod_co, on='FID_Aque')
    flopros_mod_co = flopros_mod_co[['Protection standards', 'geometry']]
    flopros_mod_co.rename({'Protection standards': 'ModL_Co'}, axis=1, inplace=True)
    flopros_mod_co.dropna(inplace=True)

    # Load population data for the year 2020 (raster = 5)
    if population_path.endswith('.zip'):
        with zipfile.ZipFile(population_path, 'r') as z:
            with z.open(z.namelist()[0]) as f:
                pop = rxr.open_rasterio(BytesIO(f.read()), masked=True).sel(raster=5)
    else:
        pop = rxr.open_rasterio(population_path, masked=True).sel(raster=5)
    pop.rio.set_crs(pop.attrs['proj4'], inplace=True)

    world_shapes = gpd.read_file(shapefiles_path, layer=0)
    world_shapes = world_shapes.rename(columns={'GID_0': 'iso3', 'ISO_A3': 'iso3'}, errors='ignore')
    world_shapes = world_shapes[['iso3', 'geometry']]
    world_shapes['iso3'] = world_shapes['iso3'].fillna('N/A')

    if world_shapes.iso3.duplicated().any():
        world_shapes = world_shapes.dissolve(by='iso3', as_index=True)
    else:
        world_shapes.set_index('iso3', inplace=True)

    protection_gridded = xr.Dataset(
        data_vars={'MerL_Riv': (('x', 'y'), np.full((len(pop.x), len(pop.y)), np.nan)),
                   'MerL_Co': (('x', 'y'), np.full((len(pop.x), len(pop.y)), np.nan))},
        coords={d: pop.coords[d].values for d in pop.dims})
    protection_gridded.rio.set_crs(pop.attrs['proj4'], inplace=True)

    # Assign modeled coastal protection levels to grid
    for i, row in tqdm.tqdm(flopros_mod_co.iterrows(), desc='assigning modeled coastal flood protection levels to grid',
                            total=len(flopros_mod_co)):
        mask = pop.rio.clip([row.geometry], flopros_mod_co.crs, drop=False).isnull()
        protection_gridded['MerL_Co'] = protection_gridded['MerL_Co'].where(mask, row['ModL_Co'])

    # Assign protection levels to grid
    for i, row in tqdm.tqdm(flopros.iterrows(), desc='assigning flopros protection levels to grid',
                            total=len(flopros)):
        mask = pop.rio.clip([row.geometry], flopros.crs, drop=False).isnull()
        protection_gridded['MerL_Riv'] = protection_gridded['MerL_Riv'].where(mask, row['MerL_Riv'])
        if not np.isnan(row['DL_PL_Co']):
            protection_gridded['MerL_Co'] = protection_gridded['MerL_Co'].where(mask, row['DL_PL_Co'])

    # Calculate population weighted protection levels
    protection_weighted = protection_gridded * pop

    prot_ntl_aggregated = gpd.GeoDataFrame(index=world_shapes.index, columns=['MerL_Riv', 'MerL_Co'])
    # Aggregate gridded protection to country level
    for iso3 in tqdm.tqdm(prot_ntl_aggregated.index, desc='aggregating to country level',
                          total=len(prot_ntl_aggregated)):
        mask = ~pop.rio.clip([world_shapes.loc[iso3, 'geometry']], world_shapes.crs, drop=False).isnull()
        country_protection = protection_weighted.where(mask).sum() / pop.where(mask).sum()
        prot_ntl_aggregated.loc[iso3, 'MerL_Riv'] = country_protection.MerL_Riv.item()
        prot_ntl_aggregated.loc[iso3, 'MerL_Co'] = country_protection.MerL_Co.item()
        prot_ntl_aggregated.loc[iso3, 'geometry'] = world_shapes.loc[iso3, 'geometry']

    if outpath is not None:
        prot_ntl_aggregated.drop('geometry', axis=1).to_csv(os.path.join(outpath, "flopros_protection_processed.csv"))

    return prot_ntl_aggregated
