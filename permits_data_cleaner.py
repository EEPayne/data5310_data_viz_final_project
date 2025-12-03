import numpy as np
import pandas as pd
import geopandas as gpd
from copy import deepcopy

def clean_permits_data(data_path, data_file_fmt = 'csv', keep_columns = None, save = False, save_fmt = 'pickle', save_path = None,
                       liquefaction_areas_path = None, slide_areas_path = None, cras_path = None):
    '''
    Hard coded cleaning and preprocessing operations for Seattle Building Permits data.
    
    Parameters
        data_path: string path or python path object
            The location of the building permits data
        data_file_fmt: string literal "json" | "csv"
            The format of the data input file
        keep_columns: list | None
            If a list of column names is given, that subset will be used instead of the default defined in the function body.
            The default is designed to work with the csv download format of the data.
        save: bool
            If true, a copy of the preprocessed data will be saved at the chosen location in the chosen format.
        save_fmt: string literal "csv" | "pickle" | "json"
            The file format to save the preprocessed data
        save_path: sting path or python path object
            The location in which to save the preprocessed data if desired.
        liquefaction_areas_path: string or path object
            liquefaction prone areas from the City of Seattle
        slide_areas_path: string or path object
            potential slide areas from the City of Seattle
        cras_path: string or path object
            Community Reporting Areas in the City of Seattle
    
    Returns
        preprocessed: Pandas.DataFrame
            Column names are ["topic", "liquefaction_prone", "slide_prone", "cra_no", "cra_name"]. A subset of the original data with selected features. A column "topic"  with values "retrofit", "damage",
            and Pandas.NA is added to designate whether an entry is confirmed to relate to a seismic retrofit or earthquake damage
    '''
    # validation

    # read in data
    if data_file_fmt == 'csv':
        data = pd.read_csv(data_path)
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

    # clean origin city name
    known_seattle_mistakes = ['seatlle']
    data['OriginalCity'] = data['OriginalCity'].str.lower()
    for mistake in known_seattle_mistakes:
        data['OriginalCity'].replace(to_replace=mistake, value='seattle', inplace=True)
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

    return data

def _add_eca_status_columns(point_data: pd.DataFrame, liquefaction_areas: gpd.GeoDataFrame, slide_areas: gpd.GeoDataFrame, cras: gpd.GeoDataFrame):
    '''
    Helper for clean_permits_data.
    Returns new point_data with an added boolean column representing whether each point is inside of any of the eca area shapes.
    values will be Pandas.NA if no shapes are given
    
    Parameters
        point_data: Pandas.DataFrame 
            The location point data
        liquefaction_areas: Geopandas.GeoDataFrame
            The ECA area shapes for liquefaction prone areas
        slide_areas: Geopandas.GeoDataFrame
            The ECA area shapes for slide prone areas (potential slide)
        cras: Geopandas.GeoDataFrame
            The Seattle Community Reporting Areas
    
    Returns
        new_data: Pandas.DataFrame
            Modified verion of the original data with the added column
    '''
    # convert points to geodataframe
    gdf_points = gpd.GeoDataFrame(
        point_data,
        geometry=gpd.points_from_xy(point_data['Longitude'], point_data['Latitude']),
        crs='EPSG:4326'   # WGS84 lat/lon
    )

    # join liquefaction areas
    print(liquefaction_areas.columns)
    join1 = gpd.sjoin(gdf_points, liquefaction_areas, how="left", predicate="within", )
    gdf_points['liquefaction_prone'] = ~join1.index_right.isna()

    # join slide areas
    print(slide_areas.columns)
    join2 = gpd.sjoin(gdf_points, slide_areas, how="left", predicate="within")
    gdf_points['slide_prone'] = ~join2.index_right.isna()

    # join community reporting areas
    print(cras.columns)
    join3 = gpd.sjoin(gdf_points, cras[['CRA_NO', 'GEN_ALIAS', 'geometry']], how="left", predicate="within")
    gdf_points['is_in_cra'] = ~join3.index_right.isna()
    gdf_points['CRA_NO'] = join3['CRA_NO']
    gdf_points['CRA_NAME'] = join3['GEN_ALIAS']

    return pd.DataFrame(gdf_points.drop(columns='geometry'))

# testing
def main():
    pass

if __name__ == '__main__':
    main()