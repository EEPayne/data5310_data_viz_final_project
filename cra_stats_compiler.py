import os
import numpy as np
import pandas as pd
import geopandas as gpd

def compile_cra_stats(seattle_census_data_path, cras_path, liquefaction_areas_path, slide_areas_path, urm_path):
    '''
    Given the most recent (2020) compiled census stats on population and housing in seattle, compile stats of interest (total housing, occupied housing, and population)
    to the level of community reporting area and return as a Pandas.DataFrame
    
    seattle_census_data_path: string or path object
        path to Annual Population and Housing Estimates for 2020 Census Blocks in Seattle data
    cras_path: string or path object
        path to Community Reporting Areas data
    liquefaction_areas_path: string or path object
        path to Liquefaction Prone Areas data
    slide_areas_path: string or path object
        Path to Potential Slide Areas Data
    urm_path: string or path object
        Path to Unreinforced Masonry (URM) Buildings dataset
    '''

    columns_to_keep = [
        'CRA_NO',                  # community reporting area id, this is the column to aggregate (sum) on
        'GEN_ALIAS',               # community reporting area name
        'CPOP_FROM_20',            # change in population from 2020
        'PCPOP_FROM_20',           # percent change in population from 2020
        'POP2024',                 # 2024 population estimate
        'POP2025',                 # 2025 population estimate
        'HU2024',                  # 2024 housing unit count
        'OHU2024',                 # 2024 occupied housing unit count
        'HU2025',                  # 2025 housing unit count
        'OHU2025'                 # 2025 occupied housing unit count
    ]

    agg_dict = {
        'CPOP_FROM_20': 'sum',
        'PCPOP_FROM_20': 'sum',
        'GEN_ALIAS': 'first',
        'POP2024': 'sum',
        'POP2025': 'sum',
        'HU2024': 'sum',
        'OHU2024': 'sum',
        'HU2024': 'sum',
        'OHU2025': 'sum'
    }

    data = gpd.read_file(seattle_census_data_path)[columns_to_keep]
    data = data.groupby(['CRA_NO']).agg(agg_dict).reset_index()
    # filter columns
    cras_land = gpd.read_file(cras_path)[['CRA_NO', 'AREA_SQMI', 'AREA_ACRES', 'WATER', 'geometry']]
    # keep only land portions of community reporting areas
    cras_land = cras_land[cras_land.WATER == 0].drop(columns=['WATER'])
    data = cras_land.merge(data, on='CRA_NO')

    # calculate eca overlap
    data = _find_eca_cra_overlaps(data, gpd.read_file(liquefaction_areas_path), 'liquefaction')
    data = _find_eca_cra_overlaps(data, gpd.read_file(slide_areas_path), 'slide')

    # count URM buildings in each CRA
    urms = gpd.read_file(urm_path).to_crs(data.crs)
    urms = gpd.sjoin(urms, data, how='left', predicate='within')
    counts = urms.groupby("index_right").size()
    data['URM_COUNT'] = data.index.map(counts).fillna(0).astype(int)

    return data

def _find_eca_cra_overlaps(cras, ecas, prefix = 'eca_overlap'):
    '''
    For each Community Reporting Area (CRA), calculates the total overlapping area with all Environmentally Critical Areas (ECA)
    as a percentage of the area of the CRA. ECAs may include liquefaction prone areas, potential slide areas, or other areas at risk in seismic events.
    *Note: EPSG:4326 represents lat/lon coordinates, and EPSG:26910 is Zone 10 of the UTM projection, in which seattle is located. UTM is good for area
     comparison within a single zone. This results initially in area in square meters, which is converted to acres and square miles to compare more easily
     with other areas in external data.
    
    Parameters
        cras: GeoDataFrame
            These are the CRAs.
        ecas: GeoDataFrame
            These are the ECAs
        prefix: string
            This will be prepended to the new column names.

    Returns
        intersected_cras: GeoDataFrame
            This will contain all the data in cras, along with three new columns containing the (total ECA overlap / CRA area) ratio,
            the overlap in acres, and the overlap in square miles. These new columns are called "_RELATIVE", "_ACRES", and "_SQ_MILES" respectively, with
            prefix prepended (i.e. "ECA_OVERLAP_RELATIVE"). The names will always be full uppercase.
    '''
    if not prefix:
        raise ValueError('Prefix cannot be empty')
    if prefix.upper() == 'AREA':
        raise ValueError('Prefix cannot be "AREA", must choose a name not in CRA dataset already.')
    original_crs = cras.crs
    intersected_cras = gpd.overlay(cras.to_crs(epsg=26910), ecas.to_crs(epsg=26910), how='intersection')
    intersected_cras['overlap_sq_meters'] = intersected_cras.geometry.area
    intersected_cras = intersected_cras.groupby(['CRA_NO'])['overlap_sq_meters'].sum().reset_index()
    intersected_cras[(prefix + '_acres').upper()] = intersected_cras['overlap_sq_meters'].apply(lambda x: x / 4046.8564224)
    intersected_cras[(prefix + '_sq_miles').upper()] = intersected_cras['overlap_sq_meters'].apply(lambda x: x / 2_589_988.110336)
    intersected_cras.drop(columns=['overlap_sq_meters'], inplace=True)
    intersected_cras = cras.merge(intersected_cras, how='left', on='CRA_NO')
    intersected_cras[(prefix + '_relative').upper()] = intersected_cras[(prefix + '_acres').upper()] / intersected_cras['AREA_ACRES']

    # revert to original crs
    return intersected_cras.to_crs(original_crs)
    