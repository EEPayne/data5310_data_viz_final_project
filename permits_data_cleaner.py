import pandas as pd
import geopandas as gpd

def clean_permits_data(data_path, data_file_fmt = 'csv', keep_columns = None, save = False, save_fmt = 'pickle'):
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
    data['EstProjectCost'] = data['EstProjectCost'].str.split(',').apply(lambda x: ''.join(x))

    # clean origin city name

    # remove rows with missing values in non-date columns

    # create "topic" column

    # save if desired
    return data



# testing
def main():
    pass

if __name__ == '__main__':
    main()