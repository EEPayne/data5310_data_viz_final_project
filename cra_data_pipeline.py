import os
import numpy as np
import pandas as pd
import geopandas as gpd

from copy import deepcopy


def _ensure_crs(gdf: gpd.GeoDataFrame, target_crs: str):
    '''
    Ensure a GeoDataFrame has the specified CRS. If gdf.crs is None, set it
    to target_crs; otherwise reproject to target_crs if different.
    '''
    if gdf is None:
        return gdf
    # If gdf has no crs, assume it's already in target_crs coordinates and set it.
    if getattr(gdf, 'crs', None) is None:
        try:
            return gdf.set_crs(target_crs)
        except Exception:
            return gdf
    # If already the same, return as-is
    try:
        if gdf.crs == target_crs or (hasattr(gdf.crs, 'to_string') and gdf.crs.to_string() == target_crs):
            return gdf
    except Exception:
        pass
    try:
        return gdf.to_crs(target_crs)
    except Exception:
        return gdf

def clean_permits_data(data_path, data_file_fmt = 'csv', keep_columns = None, save = False, save_fmt = 'pickle', save_path = None,
                       liquefaction_areas_path = None, slide_areas_path = None, cras_path = None):
    '''
    Processing for building permits data
    
    Parameters
        data_path: Path to building permits data file (csv or json)
        data_file_fmt: The format of the building permit data file ('csv' | 'json')
        keep_columns: Columns to keep if different from default
        save: If true, save the processed data to disk
        save_fmt: The format in which to save the data file ('pickle' | 'csv' | 'json')
        save_path: The location to save the processed data file
        liquefaction_areas_path: Path to ECA liquefaction prone areas geojson
        slide_areas_path: Path to ECA potential slide areas geojson
        cras_path: path to Community Reporting Areas geojson

    returns
        A processed copy of the data at data_path
    '''

    # read in data
    if data_file_fmt == 'csv':
        data = pd.read_csv(data_path, low_memory=False)
    elif data_file_fmt == 'json':
        data = pd.read_json(data_path)
    else:
        raise ValueError(f'Expected one of "csv" or "json" for data_file_fmt, got {repr(data_file_fmt)}')

    # subset columns
    if keep_columns is None:
        if data_file_fmt in ['csv', 'json']:
            keep_columns = ['PermitNum',
                            'PermitClass',
                            'PermitClassMapped',
                            'PermitTypeMapped',
                            'PermitTypeDesc',
                            'Description',
                            'EstProjectCost',
                            'AppliedDate',
                            'ReadyToIssueDate',
                            'IssuedDate',
                            'ExpiresDate',
                            'CompletedDate',
                            'StatusCurrent',
                            'OriginalAddress1',
                            'OriginalCity',
                            'OriginalState',
                            'OriginalZip',
                            'Latitude',
                            'Longitude',
                            'TotalDaysPlanReview',
                            'NumberReviewCycles',
                            'Zoning']
            
    data = data[keep_columns]

    # clean estimated project costs
    data["EstProjectCost"] = data["EstProjectCost"].astype(str).str.replace(",", "", regex=False).apply(pd.to_numeric, errors="coerce")

    # parse key dates and compute completion time (in days) where possible
    for col in ["AppliedDate", "IssuedDate", "ExpiresDate", "CompletedDate"]:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], errors="coerce")

    # CompletionTime: number of days between IssuedDate and CompletedDate (float days)
    if "IssuedDate" in data.columns and "CompletedDate" in data.columns:
        data["CompletionTime"] = (data["CompletedDate"] - data["IssuedDate"]).dt.total_seconds() / 86400.0
    else:
        # Ensure column exists even if dates not present
        data["CompletionTime"] = pd.NA

    # clean origin city name
    known_seattle_mistakes = ['seatlle']
    data['OriginalCity'] = data['OriginalCity'].str.lower()
    for mistake in known_seattle_mistakes:
        data['OriginalCity'] = data['OriginalCity'].replace(to_replace=mistake, value='seattle')
    data['OriginalCity'] = data['OriginalCity'].str.title()

    # remove rows with missing values in non-date columns
    nan_remove_cols = [
        'PermitNum',
        'Latitude',
        'Longitude',
    ]
    data.dropna(subset=nan_remove_cols, inplace=True)

    # create "topic" column
    RETROFIT_PHRASES = [
        'seismic retrofit',
        'seismic upgrade',
        'seismic proof',
        'seismic home retrofit',
        'seismic home upgrade',
        'seismic home proof',
        'seismically retrofit',
        'seismically upgrade',
        'seismically proof',
        'earthquake retrofit',
        'earthquake upgrade',
        'earthquake proof',
        'earthquake home retrofit',
        'earthquake home upgrade',
        'earthquake home proof'
    ]

    DAMAGE_PHRASES = [
        'seismic damage',
        'earthquake damage'
    ]

    # find descriptions mentioning retrofit phrases or earthquake damage and create a new topic field
    def normalize_desc(description):
        if not isinstance(description, str):
            return pd.NA
        return ' '.join(description.split()).lower()

    def categorize_topic(desc):
        if not isinstance(desc, str):
            return pd.NA
        if any([phrase in desc for phrase in RETROFIT_PHRASES]):
            return 'retrofit'
        if any([phrase in desc for phrase in DAMAGE_PHRASES]):
            return 'damage'
        return pd.NA
    
    data['topic'] = data['Description'].apply(normalize_desc).apply(categorize_topic)

    # categorize membership in ECAs and CRAs
    if not (liquefaction_areas_path is None or slide_areas_path is None or cras_path is None):
        print('Adding columns for slide risk, liquefaction risk, and community reporting areas.')
        liquefaction_areas = gpd.read_file(liquefaction_areas_path)
        slide_areas = gpd.read_file(slide_areas_path)
        cras = gpd.read_file(cras_path)
        data = _add_eca_status_columns(data, liquefaction_areas, slide_areas, cras)
    else:
        print('Did not add columns for slide risk, liquefaction risk, and community reporting areas.')

    # save if desired
    if save:
        match save_fmt:
            case 'pickle':
                data.to_pickle(save_path)
            case 'csv':
                data.to_csv(save_path)
            case 'json':
                data.to_json(save_path)

    # aliases for lat lon for convenience
    data['X'] = data['Longitude']
    data['Y'] = data['Latitude']

    return data


def _add_eca_status_columns(point_data: pd.DataFrame, liquefaction_areas: gpd.GeoDataFrame, slide_areas: gpd.GeoDataFrame, cras: gpd.GeoDataFrame):
    '''
    For each point in the point_data, compute which ECAs and CRAs the point resides in. New boolean fields ['liquefaction_prone', 'slide_prone', 'is_in_cra']
    and nominal fields ['CRA_NO', 'CRA_NAME'] are added to the point data.
    
    Parameters
        point_data: DataFrame or GeoDataFrame
            Spatial points to assign to ECAs
        liquefaction_areas: GeoDataFrame
            Liquefaction Prone Areas in Seattle
        slide_areas: GeoDataFrame
            Potential SLide Areas in Seattle
        cras: GeoDataFrame
            Community Reporting Areas
    
    Returns
        A DataFrame copy of point_data with the new fields appended.

    '''
    # convert points to geodataframe
    gdf_points = gpd.GeoDataFrame(
        point_data,
        geometry=gpd.points_from_xy(point_data['Longitude'], point_data['Latitude']),
        crs='EPSG:4326'   # WGS84 lat/lon
    )
    # Ensure CRSs match before spatial operations: use EPSG:4326 for point-in-polygon joins
    gdf_points = _ensure_crs(gdf_points, 'EPSG:4326')
    liquefaction_areas = _ensure_crs(liquefaction_areas, gdf_points.crs)
    slide_areas = _ensure_crs(slide_areas, gdf_points.crs)
    cras = _ensure_crs(cras, gdf_points.crs)

    # join liquefaction areas
    join1 = gpd.sjoin(gdf_points, liquefaction_areas, how="left", predicate="within")
    gdf_points['liquefaction_prone'] = ~join1.index_right.isna()

    # join slide areas
    join2 = gpd.sjoin(gdf_points, slide_areas, how="left", predicate="within")
    gdf_points['slide_prone'] = ~join2.index_right.isna()

    # join community reporting areas
    join3 = gpd.sjoin(gdf_points, cras[['CRA_NO', 'GEN_ALIAS', 'geometry']], how="left", predicate="within")
    gdf_points['is_in_cra'] = ~join3.index_right.isna()
    gdf_points['CRA_NO'] = join3['CRA_NO']
    gdf_points['CRA_NAME'] = join3['GEN_ALIAS']

    return pd.DataFrame(gdf_points.drop(columns='geometry'))


def clean_urm_data(urm_data_path, cras_path):
    '''
    Process URM data.
    1. Drop columns ['COMPLIANCE_METHOD', 'COUNCIL_DISTRICT', 'OVERLAY_DISTRICT', 'LANDMARK_STATUS']
    2. Drop water portions of CRAs
    3. Merge with land CRA shapes
    4. Creat LAt/Lon fields from original point geometry
    
    Parameters
        urm_data_path: Path to Unreinforced Masonry Buildings geojson
        cras_path: Path to Community Reporting Area geojson
    
    Returns
        GeoDataFrame with the merged and processed fields.
    '''
    columns_to_drop = [
        'COMPLIANCE_METHOD',
        'COUNCIL_DISTRICT',
        'OVERLAY_DISTRICT',
        'LANDMARK_STATUS'
    ]
    cras = gpd.read_file(cras_path)
    cras = _ensure_crs(cras, 'EPSG:4326')
    cras_land = cras[cras.WATER == 0][['CRA_NO', 'GEN_ALIAS', 'geometry']] if 'WATER' in cras.columns else cras[['CRA_NO', 'GEN_ALIAS', 'geometry']]
    cras_land = cras_land.rename_geometry('geometry_cra')

    urms = gpd.read_file(urm_data_path)
    try:
        print(urms.crs)
    except AttributeError as e:
        print('urms is not a geodataframe or is missing crs')
        print(type(urms).__name__)
    try:
        print(cras_land.crs)
    except AttributeError as e:
        print('cras_land is not a geodataframe or is missing crs')
        print(type(cras_land).__name__)
    urms = _ensure_crs(urms, cras_land.crs)
    urms = urms.merge(cras_land, how='left', left_on='NEIGHBORHOOD', right_on='GEN_ALIAS')
    urms = urms.drop(columns=columns_to_drop + ['GEN_ALIAS'], errors='ignore').rename(columns={'NEIGHBORHOOD': 'CRA_NAME'})
    urms['LATITUDE'] = urms.geometry.y
    urms['LONGITUDE'] = urms.geometry.x
    urms['X'] = urms.geometry.x
    urms['Y'] = urms.geometry.y
    return urms




def compile_cra_stats(seattle_census_data_path=os.path.join('data', 'OFM_SAEP_BLOCK20_ESTIMATES_SEATTLE_-7113746441103743061.geojson'),
                      cras_path=os.path.join('data', 'CITYPLAN_CRA_-6672415173103925082.geojson'),
                      liquefaction_areas_path=os.path.join('data', 'Environmentally_Critical_Areas_ECA_Liquefaction.geojson'),
                      slide_areas_path=os.path.join('data', 'Environmentally_Critical_Areas_ECA_PotentialSlide.geojson'),
                      urm_path=os.path.join('data', 'Unreinforced_Masonry_Buildings_(URM).geojson'),
                      permits_path=os.path.join('data', 'Building_Permits_20251204.csv'),
                      permits_file_fmt = 'csv'):
    '''
    Compile Statistics and the level of Community Reporting Areas (CRA) from various sources.

    Parameters    
        seattle_census_data_path: Path to census block estimate data for Seattle (geojson)
        cras_path: Path to CRA data geojson
        liquefaction_areas_path: Path to ECA liquefaction prone areas geojson
        slide_areas_path: Path to ECA potential slide areas geojson
        urm_path: Path to Unreinforced Masonry Buildings (URM) geojson
        permits_path: Path to Seattle Building permits data (likely a csv)
        permits_file_fmt: The formate of the building permits file ('csv' | 'json')

    Returns:
        data: a GeoDataFrame with CRA level statistics and polygons merged from various sources.
    '''


    census_gdf = gpd.read_file(seattle_census_data_path)

    # autoselect lates population data
    pop_cols = [c for c in census_gdf.columns if str(c).upper().startswith('POP20')]
    if len(pop_cols) > 0:
        pop_field = sorted(pop_cols)[-1]
    else:
        for candidate in ['POP2025', 'POP2024', 'POP2023']:
            if candidate in census_gdf.columns:
                pop_field = candidate
                break
        else:
            pop_field = None

    # fields to keep from census block data
    preserve_fields = [f for f in ['CPOP_FROM_20', 'PCPOP_FROM_20'] if f in census_gdf.columns]

    agg_dict = {field: 'sum' for field in preserve_fields}
    if pop_field is not None:
        agg_dict[pop_field] = 'sum'

    if 'GEN_ALIAS' in census_gdf.columns:
        agg_dict['GEN_ALIAS'] = 'first'

    if 'CRA_NO' not in census_gdf.columns:
        raise ValueError('Expected census data to contain a CRA_NO column for aggregation.')

    data = census_gdf.groupby('CRA_NO').agg(agg_dict).reset_index()

    # remove water portions of CRAs
    cras = gpd.read_file(cras_path)
    if 'WATER' in cras.columns:
        cras_land = cras[cras['WATER'] == 0].drop(columns=['WATER']).copy()
    else:
        cras_land = cras.copy()

    # compute area fields if necessary
    if 'AREA_ACRES' not in cras_land.columns or 'AREA_SQMI' not in cras_land.columns:
        tmp = cras_land.to_crs(epsg=26910)
        cras_land['AREA_SQMI'] = tmp.geometry.area.apply(lambda x: x / 2_589_988.110336)
        cras_land['AREA_ACRES'] = tmp.geometry.area.apply(lambda x: x / 4046.8564224)

    # Merge CRA polygons and area with data already aggregated form census data
    data = cras_land.merge(data, how='left', on='CRA_NO')

    # Calculate raw and relative ECA area overlap with each CRA
    if liquefaction_areas_path and os.path.exists(liquefaction_areas_path):
        data = _find_eca_cra_overlaps(data, gpd.read_file(liquefaction_areas_path), 'liquefaction')
    else:
        data['LIQUEFACTION_ACRES'] = 0.0
        data['LIQUEFACTION_SQ_MILES'] = 0.0
        data['LIQUEFACTION_RELATIVE'] = 0.0

    if slide_areas_path and os.path.exists(slide_areas_path):
        data = _find_eca_cra_overlaps(data, gpd.read_file(slide_areas_path), 'slide')
    else:
        data['SLIDE_ACRES'] = 0.0
        data['SLIDE_SQ_MILES'] = 0.0
        data['SLIDE_RELATIVE'] = 0.0

    # consolidate and merge in Unreinforced Masonry Data
    # (URM counts and risk classification)
    # Also create a risk index from URM data, which already includes information about ECAs, so including them explicitly
    # turns out to be unecessary
    if urm_path and os.path.exists(urm_path):
        urms = gpd.read_file(urm_path).to_crs(data.crs)
        urms_sjoined = gpd.sjoin(urms, data[['CRA_NO', 'geometry']], how='left', predicate='within')
        urm_counts = urms_sjoined.groupby('CRA_NO').size().reset_index(name='n_urm')
        if 'VULNERABILITY_CLASSIFICATION' in urms_sjoined.columns:
            vuln_weights = {'Medium': 1.0, 'High': 2.0, 'Critical': 3.0}
            urms_sjoined['vuln_weight'] = urms_sjoined['VULNERABILITY_CLASSIFICATION'].map(vuln_weights).fillna(0)
            vuln_agg = urms_sjoined.groupby('CRA_NO').agg(risk_weighted=('vuln_weight', 'sum'))
            urm_counts = urm_counts.merge(vuln_agg.reset_index(), on='CRA_NO', how='left')
        else:
            urm_counts['risk_weighted'] = 0.0

        for col in ['ECA_LIQUEFACTION', 'ECA_POTENTIAL_SLIDE', 'CONFIRMED_RETROFIT']:
            if col in urms_sjoined.columns:
                agg = urms_sjoined.groupby('CRA_NO').agg(**{f'n_urm_{col.lower()}': (col, lambda s: (s == 'Yes').sum())})
                urm_counts = urm_counts.merge(agg.reset_index(), on='CRA_NO', how='left')

        data = data.merge(urm_counts, on='CRA_NO', how='left')
        data['n_urm'] = data['n_urm'].fillna(0).astype(int)
        data['risk_weighted'] = data['risk_weighted'].fillna(0).astype(float)
        if 'n_urm_confirmed_retrofit' in data.columns:
            data['n_urm_confirmed_retrofit'] = data['n_urm_confirmed_retrofit'].fillna(0).astype(int)
        else:
            data['n_urm_confirmed_retrofit'] = 0
        data['n_urm_liq'] = data.get('n_urm_eca_liquefaction', 0).fillna(0).astype(int) if 'n_urm_eca_liquefaction' in data.columns else data.get('n_urm_eca_liquefaction', 0)
        data['n_urm_slide'] = data.get('n_urm_eca_potential_slide', 0).fillna(0).astype(int) if 'n_urm_eca_potential_slide' in data.columns else data.get('n_urm_eca_potential_slide', 0)
        data['n_urm_retrofit'] = data.get('n_urm_confirmed_retrofit', 0).fillna(0).astype(int)
        data['risk_score'] = data['risk_weighted'] + 0.5 * data['n_urm_liq'].fillna(0) + 0.5 * data['n_urm_slide'].fillna(0)
        data['urm_retrofit_share'] = data['n_urm_retrofit'] / data['n_urm'].replace({0: np.nan})
    else:
        data['n_urm'] = 0
        data['risk_weighted'] = 0.0
        data['n_urm_liq'] = 0
        data['n_urm_slide'] = 0
        data['n_urm_retrofit'] = 0
        data['risk_score'] = 0.0
        data['urm_retrofit_share'] = np.nan

    # calculate Permit count and Retrofit permit count by CRA, find relative retrofits as a share of total permits,
    # and find retrofit intensity as retrofits / 10K population in each CRA
    if permits_path and os.path.exists(permits_path):
        permits = clean_permits_data(permits_path, permits_file_fmt, liquefaction_areas_path=liquefaction_areas_path,
                                     slide_areas_path=slide_areas_path, cras_path=cras_path)
        data = _add_cra_permit_counts(data, permits)
    else:
        data['BLDG_PERMIT_COUNT'] = 0
        data['RETROFIT_PERMIT_COUNT'] = 0

    pop_col = None
    for candidate in ['POP2025', 'POP2024'] + [c for c in data.columns if str(c).upper().startswith('POP20')]:
        if candidate in data.columns:
            pop_col = candidate
            break
    if pop_col is None:
        pop_col = 'population'

    if pop_col in data.columns:
        data['population'] = data[pop_col]
    else:
        data['population'] = np.nan

    data['retrofit_share_permits'] = data['RETROFIT_PERMIT_COUNT'] / data['BLDG_PERMIT_COUNT'].replace({0: np.nan})
    data['retrofit_rate_per_10k'] = data['RETROFIT_PERMIT_COUNT'] / data['population'].replace({0: np.nan}) * 10000

    # use this to normalize risk and mitigation indices
    def minmax(s):
        s = s.astype(float)
        if s.max() == s.min():
            return s * 0.0
        return (s - s.min()) / (s.max() - s.min())

    # risk index from risk score, mitigation index from URM and permit retrofits
    data['risk_index'] = minmax(data['risk_score'].fillna(0))
    data['mitigation_index'] = minmax((data['retrofit_rate_per_10k'].fillna(0) + data['urm_retrofit_share'].fillna(0)))

    # ensure only one version of CRA name
    if 'GEN_ALIAS' not in data.columns:
        if 'GEN_ALIAS_y' in data.columns and 'GEN_ALIAS_x' in data.columns:
            data['GEN_ALIAS'] = data['GEN_ALIAS_y'].fillna(data['GEN_ALIAS_x'])
            data = data.drop(columns=['GEN_ALIAS_x', 'GEN_ALIAS_y'])
        elif 'GEN_ALIAS_y' in data.columns:
            data['GEN_ALIAS'] = data['GEN_ALIAS_y']
            data = data.drop(columns=['GEN_ALIAS_y'])
        elif 'GEN_ALIAS_x' in data.columns:
            data['GEN_ALIAS'] = data['GEN_ALIAS_x']
            data = data.drop(columns=['GEN_ALIAS_x'])

    # add geometry back in for GeoDataFrame if necessary
    try:
        if 'geometry' not in data.columns and data.geometry.name != 'geometry':
            data = data.set_geometry(data.geometry)
    except Exception:
        pass

    return data


def _find_eca_cra_overlaps(cras, ecas, prefix = 'eca_overlap'):
    '''
    Docstring for _find_eca_cra_overlaps
    
    Parameters
        cras: Clean data defining CRA polygons
        ecas: Clean data defining ECA polygons (likely either potential slide areas or liquefaction prone areas)
        prefix: String to prepend to the new column name
    
    Returns
        A copy of the cras data, with a three new columns appended named [prefix+'_ACRES', prefix+'_SQ_MILES', prefix+'_RELATIVE']
        representing the total overlap between a given CRA and all ECA polygons given.
    '''

    if not prefix:
        raise ValueError('Prefix cannot be empty')
    if prefix.upper() == 'AREA':
        raise ValueError('Prefix cannot be "AREA", must choose a name not in CRA dataset already.')
    original_crs = cras.crs

    # Project both inputs to a projected CRS suitable for area (UTM / state plane)
    cras_proj = cras.to_crs(epsg=26910).copy()
    ecas_proj = ecas.to_crs(epsg=26910).copy()

    # Fix invalid geometries where possible (buffer(0) is a common trick)
    try:
        cras_proj['geometry'] = cras_proj.geometry.buffer(0)
    except Exception:
        pass
    try:
        ecas_proj['geometry'] = ecas_proj.geometry.buffer(0)
    except Exception:
        pass

    # Attempt overlay; if overlay fails or returns empty, return CRAs with zero overlap columns
    try:
        intersected = gpd.overlay(cras_proj, ecas_proj, how='intersection')
    except Exception:
        intersected = gpd.GeoDataFrame(columns=list(cras_proj.columns) + ['overlap_sq_meters'], geometry=[])

    if intersected.empty:
        # Ensure the CRA frame has the expected output columns (uppercase names used elsewhere)
        out = cras.copy()
        out[(prefix + '_ACRES').upper()] = 0.0
        out[(prefix + '_SQ_MILES').upper()] = 0.0
        out[(prefix + '_RELATIVE').upper()] = 0.0
        # Keep original CRS
        try:
            return out.to_crs(original_crs)
        except Exception:
            return out

    # Compute intersection areas (in square meters because CRS is projected)
    intersected['overlap_sq_meters'] = intersected.geometry.area
    summed = intersected.groupby('CRA_NO', as_index=False)['overlap_sq_meters'].sum()
    acres_col = (prefix + '_acres').upper()
    sqmi_col = (prefix + '_sq_miles').upper()
    summed[acres_col] = summed['overlap_sq_meters'] / 4046.8564224
    summed[sqmi_col] = summed['overlap_sq_meters'] / 2_589_988.110336
    summed = summed.drop(columns=['overlap_sq_meters'])

    # Merge computed overlaps back onto the original CRA frame and compute relative share
    merged = cras.merge(summed, how='left', on='CRA_NO')
    # Fill missing values with zeros for numeric overlap fields
    merged[acres_col] = merged.get(acres_col, 0.0).fillna(0.0)
    merged[sqmi_col] = merged.get(sqmi_col, 0.0).fillna(0.0)

    rel_col = (prefix + '_relative').upper()
    # Avoid divide-by-zero: if AREA_ACRES is missing or zero, treat relative as 0
    merged[rel_col] = 0.0
    if 'AREA_ACRES' in merged.columns:
        with np.errstate(invalid='ignore', divide='ignore'):
            merged[rel_col] = (merged[acres_col] / merged['AREA_ACRES']).fillna(0.0)

    # If the source CRA layer contains multiple parts with the same CRA_NO
    # (multi-polygons or split geometries), the merge above can produce
    # multiple rows per CRA_NO. In most following analyses we want one
    # CRA-level row. Drop duplicate CRA_NO rows, keeping the first geometry
    # (which will preserve the computed overlap columns added above).
    try:
        merged = merged.sort_values('CRA_NO').drop_duplicates(subset='CRA_NO', keep='first')
    except Exception:
        pass

    try:
        return merged.to_crs(original_crs)
    except Exception:
        return merged


def _add_cra_permit_counts(cras, permits):
    '''
    Append CRA level statistics related to Building Permits to the cras data set. Added fields include the count of Building permits by CRA ('BLDG_PERMIT_COUNT'),
    count of retrofit permits ('RETROFIT_PERMIT_COUNT'), mean estimated project cost ('AVG_EST_PROJECT_COST'),
    and mean prokect completion time in days for completed projects ('AVG_COMPLETION_TIME_DAYS')
    
    Parameters
        cras: Clean CRA GeoDataFrame
        permits: Clean Building Permits DataFrame

    returns:
        a copy of cras with the new fields appended.
    '''
    # Drop permits with no CRA association (they won't merge onto cras)
    permits_with_cra = permits.dropna(subset=['CRA_NO']).copy()

    # Aggregate counts and averages per CRA
    if 'PermitNum' in permits_with_cra.columns:
        count_agg = permits_with_cra.groupby('CRA_NO', as_index=False).agg(
            BLDG_PERMIT_COUNT=('PermitNum', 'count'),
            AVG_EST_PROJECT_COST=('EstProjectCost', 'mean'),
            AVG_COMPLETION_TIME_DAYS=('CompletionTime', 'mean')
        )
    else:
        # fallback to size if PermitNum missing
        count_agg = permits_with_cra.groupby('CRA_NO', as_index=False).agg(
            BLDG_PERMIT_COUNT=('CRA_NO', 'size'),
            AVG_EST_PROJECT_COST=('EstProjectCost', 'mean'),
            AVG_COMPLETION_TIME_DAYS=('CompletionTime', 'mean')
        )

    retrofits_by_cra = (
        permits_with_cra[permits_with_cra.topic == 'retrofit']
        .groupby('CRA_NO', as_index=False)
        .agg(RETROFIT_PERMIT_COUNT=('topic', 'size'))
    )

    new_cras = cras.merge(count_agg, how='left', on='CRA_NO').merge(retrofits_by_cra, how='left', on='CRA_NO')

    # Fill counts with zeros; keep averages as NaN when not available
    new_cras['BLDG_PERMIT_COUNT'] = new_cras['BLDG_PERMIT_COUNT'].fillna(0).astype(int)
    new_cras['RETROFIT_PERMIT_COUNT'] = new_cras['RETROFIT_PERMIT_COUNT'].fillna(0).astype(int)

    # Ensure average columns exist (float) even if there were no permits
    if 'AVG_EST_PROJECT_COST' not in new_cras.columns:
        new_cras['AVG_EST_PROJECT_COST'] = np.nan
    if 'AVG_COMPLETION_TIME_DAYS' not in new_cras.columns:
        new_cras['AVG_COMPLETION_TIME_DAYS'] = np.nan

    return new_cras
