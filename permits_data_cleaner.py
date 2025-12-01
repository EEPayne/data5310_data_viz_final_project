import pandas as pd
import geopandas as pd

def clean_permits_data(data_path, data_file_fmt = 'json', save = False, save_fmt = 'pickle'):
    '''
    Hard coded cleaning and preprocessing operations for Seattle Building Permits data.
    
    Parameters
        data_path: string path or python path object
            The location of the building permits data
        data_file_fmt: string literal "json" | "csv"
            The format of the data input file
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

    # subset columns

    # clean estimated project costs

    # clean origin city name

    # remove rows with missing values in non-date columns

    # create "topic" column

    # save if desired




# testing
def main():
    pass

if __name__ == '__main__':
    main()