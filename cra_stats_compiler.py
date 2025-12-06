import os
import numpy as np
import pandas as pd
import geopandas as gpd
from permits_data_cleaner import clean_permits_data

def compile_cra_stats(seattle_census_data_path=os.path.join('data', 'OFM_SAEP_BLOCK20_ESTIMATES_SEATTLE_-7113746441103743061.geojson'),
                      cras_path=os.path.join('data', 'CITYPLAN_CRA_-6672415173103925082.geojson'),
                      liquefaction_areas_path=os.path.join('data', 'Environmentally_Critical_Areas_ECA_Liquefaction.geojson'),
                      slide_areas_path=os.path.join('data', 'Environmentally_Critical_Areas_ECA_PotentialSlide.geojson'),
                      urm_path=os.path.join('data', 'Unreinforced_Masonry_Buildings_(URM).geojson'),
                      permits_path=os.path.join('data', 'Building_Permits_20251204.csv'),
                      permits_file_fmt = 'csv'):
    '''
    Given the most recent compiled census block estimates for Seattle, aggregate and compile a comprehensive
    CRA-level GeoDataFrame with population, permit counts, URM summaries, ECA overlap measures, and
    simple risk/mitigation indices.

    Default paths point to files in the project's `data/` folder. The function expects the census input
    to already include a `CRA_NO` identifier for each block (as in the project's census block file).

    Behavior notes (hardcoded):
    - Area/overlap calculations are done in EPSG:26910 (UTM zone 10) and converted to acres/sq miles.
    - Geometries are returned in the original CRS (typically EPSG:4326) for mapping.
    - The latest population field with prefix 'POP20' is autodetected when present; the function also
      preserves the existing growth-from-2020 fields (`CPOP_FROM_20`, `PCPOP_FROM_20`) when available.

    Parameters
    ----------
    seattle_census_data_path: str | path (default: 'data/CENSUS_BLOCK_ESTIMATES_SEATTLE.geojson')
        Block-level population/housing estimates containing `CRA_NO` to aggregate by.
    cras_path: str | path (default: 'data/seattle_community_reporting_areas.geojson')
        CRA polygons (geometry + AREA_ACRES/AREA_SQMI expected fields).
    liquefaction_areas_path: str | path (default: 'data/ECA_LiquefactionProneAreas.geojson')
        Liquefaction-prone ECA polygons.
    slide_areas_path: str | path (default: 'data/ECA_PotentialSlideAreas.geojson')
        Potential slide ECA polygons.
    urm_path: str | path (default: 'data/Unreinforced_Masonry_Buildings_(URM).geojson')
        URM building point dataset (optional columns used if present: VULNERABILITY_CLASSIFICATION,
        ECA_LIQUEFACTION, ECA_POTENTIAL_SLIDE, CONFIRMED_RETROFIT, OBJECTID).
    permits_path: str | path (default: 'data/Building_Permits_20251125.csv')
        Building permits file; passed to `permits_data_cleaner.clean_permits_data` for cleaning and CRA assignment.
    permits_file_fmt: str
        'csv' or 'json' for permits file format.

    Returns
    -------
    GeoDataFrame
        CRA-level GeoDataFrame with standardized columns including:
        - `population` (detected POP20* field or POP2024/POP2025)
        - `BLDG_PERMIT_COUNT`, `RETROFIT_PERMIT_COUNT`, `retrofit_share_permits`, `retrofit_rate_per_10k`
        - `URM` counts and risk-related fields: `n_urm`, `risk_score`, `urm_retrofit_share`, `risk_index`, `mitigation_index`
        - ECA overlap measures like `LIQUEFACTION_ACRES`, `SLIDE_ACRES`, and relative shares where present.
    '''

    # Read block-level census data and aggregate to CRA level. The census file in this repo already
    # contains a `CRA_NO` column on each block, so we can aggregate directly. We autodetect the
    # latest population field beginning with 'POP20' if present, otherwise fall back to POP2024/POP2025.
    census_gdf = gpd.read_file(seattle_census_data_path)

    # detect population field like the notebook (latest POP20*)
    pop_cols = [c for c in census_gdf.columns if str(c).upper().startswith('POP20')]
    if len(pop_cols) > 0:
        pop_field = sorted(pop_cols)[-1]
    else:
        # fallback common field names
        for candidate in ['POP2025', 'POP2024', 'POP2023']:
            if candidate in census_gdf.columns:
                pop_field = candidate
                break
        else:
            pop_field = None

    # fields to preserve if present
    preserve_fields = [f for f in ['CPOP_FROM_20', 'PCPOP_FROM_20'] if f in census_gdf.columns]

    agg_dict = {field: 'sum' for field in preserve_fields}
    if pop_field is not None:
        agg_dict[pop_field] = 'sum'

    # always keep GEN_ALIAS if present
    if 'GEN_ALIAS' in census_gdf.columns:
        agg_dict['GEN_ALIAS'] = 'first'

    if 'CRA_NO' not in census_gdf.columns:
        raise ValueError('Expected census data to contain a CRA_NO column for aggregation.')

    data = census_gdf.groupby('CRA_NO').agg(agg_dict).reset_index()

    # read CRA polygons and keep land-only portions if WATER column exists
    cras = gpd.read_file(cras_path)
    if 'WATER' in cras.columns:
        cras_land = cras[cras['WATER'] == 0].drop(columns=['WATER']).copy()
    else:
        cras_land = cras.copy()

    # ensure expected area fields exist; if not, compute from geometry in projected CRS
    if 'AREA_ACRES' not in cras_land.columns or 'AREA_SQMI' not in cras_land.columns:
        tmp = cras_land.to_crs(epsg=26910)
        cras_land['AREA_SQMI'] = tmp.geometry.area.apply(lambda x: x / 2_589_988.110336)
        cras_land['AREA_ACRES'] = tmp.geometry.area.apply(lambda x: x / 4046.8564224)

    # merge aggregated census stats onto CRA polygons
    data = cras_land.merge(data, how='left', on='CRA_NO')

    # calculate eca overlap
    # compute overlaps with ECAs (liquefaction and slide). The helper projects to EPSG:26910 for accurate area.
    if liquefaction_areas_path and os.path.exists(liquefaction_areas_path):
        data = _find_eca_cra_overlaps(data, gpd.read_file(liquefaction_areas_path), 'liquefaction')
    else:
        # create empty columns for consistency
        data['LIQUEFACTION_ACRES'] = 0.0
        data['LIQUEFACTION_SQ_MILES'] = 0.0
        data['LIQUEFACTION_RELATIVE'] = 0.0

    if slide_areas_path and os.path.exists(slide_areas_path):
        data = _find_eca_cra_overlaps(data, gpd.read_file(slide_areas_path), 'slide')
    else:
        data['SLIDE_ACRES'] = 0.0
        data['SLIDE_SQ_MILES'] = 0.0
        data['SLIDE_RELATIVE'] = 0.0

    # URM processing: spatial join URM points to CRA and aggregate vulnerability / ECA flags
    if urm_path and os.path.exists(urm_path):
        urms = gpd.read_file(urm_path).to_crs(data.crs)
        urms_sjoined = gpd.sjoin(urms, data[['CRA_NO', 'geometry']], how='left', predicate='within')
        # count URMs per CRA
        urm_counts = urms_sjoined.groupby('CRA_NO').size().reset_index(name='n_urm')
        # vulnerability weights (not all datasets have the field; map safely)
        if 'VULNERABILITY_CLASSIFICATION' in urms_sjoined.columns:
            vuln_weights = {'Medium': 1.0, 'High': 2.0, 'Critical': 3.0}
            urms_sjoined['vuln_weight'] = urms_sjoined['VULNERABILITY_CLASSIFICATION'].map(vuln_weights).fillna(0)
            vuln_agg = urms_sjoined.groupby('CRA_NO').agg(risk_weighted=('vuln_weight', 'sum'))
            urm_counts = urm_counts.merge(vuln_agg.reset_index(), on='CRA_NO', how='left')
        else:
            urm_counts['risk_weighted'] = 0.0

        # ECA flags on URMs if present
        for col in ['ECA_LIQUEFACTION', 'ECA_POTENTIAL_SLIDE', 'CONFIRMED_RETROFIT']:
            if col in urms_sjoined.columns:
                agg = urms_sjoined.groupby('CRA_NO').agg(**{f'n_urm_{col.lower()}': (col, lambda s: (s == 'Yes').sum())})
                urm_counts = urm_counts.merge(agg.reset_index(), on='CRA_NO', how='left')

        # merge back to CRA table
        data = data.merge(urm_counts, on='CRA_NO', how='left')
        data['n_urm'] = data['n_urm'].fillna(0).astype(int)
        data['risk_weighted'] = data['risk_weighted'].fillna(0).astype(float)
        # normalize optional cols
        if 'n_urm_confirmed_retrofit' in data.columns:
            data['n_urm_confirmed_retrofit'] = data['n_urm_confirmed_retrofit'].fillna(0).astype(int)
        else:
            data['n_urm_confirmed_retrofit'] = 0
        # compute simple URM-derived risk score
        data['n_urm_liq'] = data.get('n_urm_eca_liquefaction', 0).fillna(0).astype(int) if 'n_urm_eca_liquefaction' in data.columns else data.get('n_urm_eca_liquefaction', 0)
        data['n_urm_slide'] = data.get('n_urm_eca_potential_slide', 0).fillna(0).astype(int) if 'n_urm_eca_potential_slide' in data.columns else data.get('n_urm_eca_potential_slide', 0)
        data['n_urm_retrofit'] = data.get('n_urm_confirmed_retrofit', 0).fillna(0).astype(int)
        data['risk_score'] = data['risk_weighted'] + 0.5 * data['n_urm_liq'].fillna(0) + 0.5 * data['n_urm_slide'].fillna(0)
        data['urm_retrofit_share'] = data['n_urm_retrofit'] / data['n_urm'].replace({0: np.nan})
    else:
        # no URM data available: add zeroed columns
        data['n_urm'] = 0
        data['risk_weighted'] = 0.0
        data['n_urm_liq'] = 0
        data['n_urm_slide'] = 0
        data['n_urm_retrofit'] = 0
        data['risk_score'] = 0.0
        data['urm_retrofit_share'] = np.nan

    # add permit counts
    # Permits: use existing cleaner which also optionally annotates ECA/CRA membership
    if permits_path and os.path.exists(permits_path):
        permits = clean_permits_data(permits_path, permits_file_fmt, liquefaction_areas_path=liquefaction_areas_path,
                                     slide_areas_path=slide_areas_path, cras_path=cras_path)
        data = _add_cra_permit_counts(data, permits)
    else:
        data['BLDG_PERMIT_COUNT'] = 0
        data['RETROFIT_PERMIT_COUNT'] = 0

    # compute retrofit share and rate per 10k residents using the detected population field
    pop_col = None
    for candidate in ['POP2025', 'POP2024'] + [c for c in data.columns if str(c).upper().startswith('POP20')]:
        if candidate in data.columns:
            pop_col = candidate
            break
    if pop_col is None:
        # if earlier detection succeeded, use that name; otherwise fall back to explicit population column
        pop_col = 'population'

    # ensure a consistent `population` column for downstream usage
    if pop_col in data.columns:
        data['population'] = data[pop_col]
    else:
        data['population'] = np.nan

    data['retrofit_share_permits'] = data['RETROFIT_PERMIT_COUNT'] / data['BLDG_PERMIT_COUNT'].replace({0: np.nan})
    data['retrofit_rate_per_10k'] = data['RETROFIT_PERMIT_COUNT'] / data['population'].replace({0: np.nan}) * 10000

    # compute min-max scaled indices for risk and mitigation
    def minmax(s):
        s = s.astype(float)
        if s.max() == s.min():
            return s * 0.0
        return (s - s.min()) / (s.max() - s.min())

    data['risk_index'] = minmax(data['risk_score'].fillna(0))
    data['mitigation_index'] = minmax((data['retrofit_rate_per_10k'].fillna(0) + data['urm_retrofit_share'].fillna(0)))

    return data

def _find_eca_cra_overlaps(cras, ecas, prefix = 'eca_overlap'):
    '''
    Helper for compile_cra_stats
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
    

def _add_cra_permit_counts(cras, permits):
    '''
    Helper for compile_cra_stats
    Counts the total number of permits and the number of permits mentioning retrofits per CRA
    
    Parameters
        cras: GeoDataFrame
            Community Reporting Areas
        permits: GeoDataFrame
            Permits data processed with permits_data_cleaner.clean_permits_data
    
    Returns
        cras: GeoDataFrame
            copy of the original cras data with columns added for the total number of permits and the number of permits mentioning seismic retrofits.
            New columns are named "BLDG_PERMIT_COUNT" and "RETROFIT_PERMIT_COUNT".
    '''
    permits_by_cra = permits.groupby(['CRA_NO']).size().reset_index(name='BLDG_PERMIT_COUNT')
    retrofits_by_cra = permits[permits.topic == 'retrofit'].groupby(['CRA_NO']).size().reset_index(name='RETROFIT_PERMIT_COUNT')

    new_cras = cras.merge(permits_by_cra, how='left', on='CRA_NO').merge(retrofits_by_cra, how='left', on='CRA_NO')

    # fill na with 0
    new_cras.loc[:, ['BLDG_PERMIT_COUNT', 'RETROFIT_PERMIT_COUNT']] = new_cras[['BLDG_PERMIT_COUNT', 'RETROFIT_PERMIT_COUNT']].fillna(0).apply(lambda s: s.astype(int))
    
    return new_cras