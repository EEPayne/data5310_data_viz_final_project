import numpy as np
import pandas as pd
import geopandas as gpd

def clean_urm_data(urm_data_path, cras_path):
    columns_to_drop = [
        'COMPLIANCE_METHOD',
        'COUNCIL_DISTRICT',
        'OVERLAY_DISTRICT',
        'LANDMARK_STATUS'
    ]
    urms = gpd.read_file(urm_data_path)
    cras = gpd.read_file(cras_path)
    cras = cras[cras.WATER == 0][['CRA_NO', 'GEN_ALIAS']]
    urms = urms.merge(cras, how='left', left_on='NEIGHBORHOOD', right_on='GEN_ALIAS')
    urms = urms.drop(columns=columns_to_drop+['GEN_ALIAS']).rename(columns={'NEIGHBORHOOD':'CRA_NAME'})
    urms['LATITUDE'] = urms.geometry.y
    urms['LONGITUDE'] = urms.geometry.x
    return urms