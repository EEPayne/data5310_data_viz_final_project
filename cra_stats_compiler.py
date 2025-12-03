import os
import numpy as np
import pandas as pd
import geopandas as gpd

def compile_cra_stats(seattle_census_data_path, cras_path):
    '''
    Given the most recent (2020) compiled census stats on population and housing in seattle, compile stats of interest (total housing, occupied housing, and population)
    to the level of community reporting area and return as a Pandas.DataFrame
    
    Parameters
        seattle_census_data_path: Description
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
        'OHU2025',                 # 2025 occupied housing unit count
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

    cras = gpd.read_file(cras_path)[['CRA_NO', 'geometry']].merge(data, on='CRA_NO', suffixes=('', '_right'))

    return cras
    