import pandas as pd

def clean_permits_data(data_path, data_file_fmt = 'csv', keep_columns = None, save = False, save_fmt = 'pickle', save_path = None):
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
    
    Returns
        preprocessed: Pandas.DataFrame
            Column names are []. A subset of the original data with selected features. A column "topic"  with values "retrofit", "damage",
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
    data['OriginalCity'] = data['OriginalCity'].str.capitalize()

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



# testing
def main():
    pass

if __name__ == '__main__':
    main()