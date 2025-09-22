import pyreadr
from pathlib import Path
import pandas as pd
import shutil
from IPython.display import display


class TEPDataLoader:
    """
    A data loader class for Tennessee Eastman Process (TEP) simulation datasets.

    This class provides functionality to load, convert, and manage TEP datasets which are
    commonly used for process monitoring and fault detection research. The TEP dataset
    contains both normal and faulty operation scenarios for industrial process simulation.
    The class handles R data files (.RData format) and converts them to CSV format for
    easier manipulation in Python environments.

    The Tennessee Eastman Process is a well-known benchmark simulation that represents
    a realistic industrial chemical process with multiple operating modes and fault
    scenarios. This makes it invaluable for testing process monitoring algorithms
    and fault detection systems.

    Attributes:
        raw_data_path (Path): Path to the directory containing raw .RData files.
        processed_data_path (Path): Path to the directory where processed CSV files
            will be saved. Directory is created if it doesn't exist.
        file_info (dict): Dictionary containing metadata about each TEP file, including
            file type classification and human-readable descriptions.
    """

    fault_descriptions = {
        0: 'Normal operation',
        1: 'A/C feed ratio, B composition constant (step change in stream 4)',
        2: 'B composition, A/C ratio constant (stream 4)',
        3: 'D feed temperature (stream 2)',
        4: 'Reactor cooling water inlet temperature',
        5: 'Condenser cooling water inlet temperature',
        6: 'A feed loss (step change in stream 1)',
        7: 'C header pressure loss—reduced availability (stream 4)',
        8: 'A, B, C feed composition (stream 4)',
        9: 'D feed temperature (stream 2)',
        10: 'C feed temperature (stream 4)',
        11: 'Reactor cooling water inlet temperature',
        12: 'Condenser cooling water inlet temperature',
        13: 'Reaction kinetics (slow drift)',
        14: 'Reactor cooling water valve',
        15: 'Condenser cooling water valve',
        16: 'Unknown disturbances',
        17: 'Unknown disturbances',
        18: 'Unknown disturbances',
        19: 'Unknown disturbances',
        20: 'Unknown disturbances',
    }

    def __init__(self, raw_data_path: str, processed_data_path: str):
        """
        Initializes the TEP data loader with input and output directory paths.

        This constructor sets up the file paths and creates the output directory
        structure. It also defines the metadata for all expected TEP data files,
        categorizing them by their purpose (training vs testing) and condition
        (normal vs faulty operation).

        Args:
            raw_data_path (str): Path to directory containing the original .RData files.
                This should contain the four standard TEP files from the simulation.
            processed_data_path (str): Path to directory where converted CSV files
                will be saved. Directory will be created if it doesn't exist.
        """
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path)

        self.processed_data_path.mkdir(exist_ok=True)

        self.file_info = {  # Basic info about files
            'TEP_FaultFree_Training.RData': {
                'type': 'fault_free_training',
                'description': 'Normal process operation data for training'
            },
            'TEP_FaultFree_Testing.RData': {
                'type': 'fault_free_testing',
                'description': 'Normal process operation data for testing'
            },
            'TEP_Faulty_Training.RData': {
                'type': 'faulty_training',
                'description': 'Faulty process operation data for training'
            },
            'TEP_Faulty_Testing.RData': {
                'type': 'faulty_testing',
                'description': 'Faulty process operation data for testing'
            }
        }

    def load_rdata_file(self, filename):
        """
        Loads a single R data file and returns it as a pandas DataFrame.

        This method handles the conversion from R's native data format to Python's
        pandas DataFrame. R data files can contain multiple objects, but TEP files
        typically contain a single dataset which this method extracts automatically.

        Args:
            filename (str): Name of the .RData file to load from the raw data directory.
                Should be one of the standard TEP filenames.

        Returns:
            pd.DataFrame: The loaded dataset as a pandas DataFrame with all original
            columns and data types preserved from the R format.

        Note:
            Uses pyreadr library which requires R to be installed on the system.
            The method assumes the .RData file contains exactly one dataset object.
        """
        file_path = self.raw_data_path / filename

        result = pyreadr.read_r(str(file_path))

        data_key = list(result.keys())[0]

        return result[data_key]


    def convert_and_save_to_csv(self):
        """
        Converts all TEP .RData files to CSV format and saves them to the processed directory.

        This method processes each file defined in the file_info dictionary, loading
        the R data format and converting it to CSV for easier use in Python workflows.
        The converted files are given standardized names based on their data type
        rather than the original filenames.

        The CSV format is more universally accessible and doesn't require R dependencies,
        making the data easier to work with in various Python environments and tools.

        Returns:
            None: Files are saved to disk but no data is returned.

        Note:
            - Original DataFrames are deleted after saving to manage memory usage
            - CSV files are saved without row indices to maintain clean data structure
            - File naming follows the pattern: TEP_{data_type}.csv
        """
        for filename, info in self.file_info.items():
            df = self.load_rdata_file(filename)

            csv_filename = f"TEP_{info['type']}.csv"
            csv_path = self.processed_data_path / csv_filename

            df.to_csv(csv_path, index=False)

            del df

        print('All files have been saved!')


def filter_csv(filename: str,
               selected_faults: list,
               max_simulations: int,
               data_path: str) -> None:
    """
    Filters a CSV file based on fault numbers and simulation run limits, saving the result.

    This function is particularly useful for working with Tennessee Eastman Process (TEP)
    datasets or similar simulation data where you need to focus on specific fault
    scenarios and limit the amount of data for analysis or model training. The filtering
    helps reduce dataset size while maintaining focus on relevant fault conditions.

    The function operates by creating a boolean mask that combines two conditions:
    the fault number must be in the selected list AND the simulation run number
    must not exceed the specified maximum. This approach allows for efficient
    filtering of large datasets without loading unnecessary data into memory.

    Args:
        filename (str): Name of the CSV file to filter (without .csv extension).
            The function will automatically append '.csv' for reading and
            '_filtered.csv' for the output file.
        selected_faults (list): List of fault numbers to include in the filtered dataset.
            These should correspond to the fault identifiers in the 'faultNumber' column
            of the source data. For example, [0, 1, 5] would include normal operation
            (fault 0) and faults 1 and 5.
        max_simulations (int): Maximum simulation run number to include. Simulation
            runs with numbers greater than this value will be excluded from the
            filtered dataset. This helps limit dataset size and focus on specific
            simulation scenarios.
        data_path (str): Path to the directory containing the input CSV file and
            where the filtered output file will be saved.

    Returns:
        None: The function saves the filtered data to a new CSV file but does not
        return any value.

    Note:
        - Input file format: {data_path}/{filename}.csv
        - Output file format: {data_path}/{filename}_filtered.csv
        - Requires 'faultNumber' and 'simulationRun' columns in the source data
        - Original DataFrame is deleted after saving to manage memory usage
        - Output file excludes row indices for cleaner data structure
    """

    df = pd.read_csv(f'{data_path}/{filename}.csv')

    mask = ((df['faultNumber'].isin(selected_faults)) &
            (df['simulationRun'] <= max_simulations))

    df = df[mask]
    df.to_csv(f'{data_path}/{filename}_filtered.csv', index=False)

    del df




def describe_dataframe(df: pd.DataFrame, head: int = 5, delete_duplicates: bool = False) -> pd.DataFrame:
    """
    Prints a quick overview of the provided DataFrame for exploratory data analysis (EDA).

    This function displays the first rows, shape, basic statistics, missing values,
    number of duplicates, and unique values per column. Optionally, it can remove duplicate rows
    and return the cleaned DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame to analyze.
        head (int, optional): The number of top rows to display. Defaults to 5.
        delete_duplicates (bool, optional): Whether to remove duplicate rows from the DataFrame. Defaults to False.

    Returns:
        pd.DataFrame: A copy of the input DataFrame, possibly with duplicates removed.

    Example:
        >>> cleaned_df = describe_dataframe(df, head=10, delete_duplicates=True)
    """

    def rows_separator():
        try:
            w = shutil.get_terminal_size().columns
        except:
            w = 100

        print('_' * w, end='\n\n')

    print(f'First {head} rows of the DataFrame:')
    display(df.head(head))
    rows_separator()

    print('Shape of the DataFrame:', df.shape)
    rows_separator()

    print('DataFrame info:')
    df.info()
    rows_separator()

    print('Descriptive statistics of the dataframe:')
    display(df.describe().T)
    rows_separator()

    print('Missing values per column:')
    display(df.isna().sum())
    print('Total missing values:', df.isna().sum().sum())
    rows_separator()

    p = round(df.duplicated().sum()/df.shape[0]*100, 3)
    print(f'Duplicate rows: {p}%', )
    rows_separator()

    if delete_duplicates:
        if p:
            df = df.drop_duplicates(inplace=False, ignore_index=False)
            print('Duplicate rows removed.')
        else:
            print('No duplicate rows to remove.')
        rows_separator()

    print('Unique values per column:')
    display(df.nunique())

    return df


def get_feature_description(feature_name: str, descriptions: dict) -> str:
    """
    Converts engineered feature names into human-readable descriptions.

    This function transforms technical feature names (especially those created through
    feature engineering processes) into descriptive, human-readable text that explains
    what each feature represents. It handles various types of engineered features
    including rolling statistics, lag features, and difference features by parsing
    the feature name structure and combining base descriptions with transformation
    information.

    The function is particularly useful for model interpretation, feature importance
    visualization, and generating reports where technical feature names need to be
    presented in a more accessible format for stakeholders or documentation purposes.

    Args:
        feature_name (str): The technical feature name to convert. Should follow
            standard naming conventions like 'base_feature_transformation_parameter'
            (e.g., 'XMEAS_1_rolling_mean_5', 'XMEAS_2_lag_3', 'XMEAS_3_diff').
        descriptions (dict): Dictionary mapping base feature names to their
            human-readable descriptions. Keys should match the base portion
            of engineered feature names (e.g., 'XMEAS_1' -> 'Reactor Feed Rate').

    Returns:
        str: Human-readable description of the feature including both the base
        variable description and the transformation applied. For example:
        - 'XMEAS_1_rolling_mean_5' -> 'Reactor Feed Rate (rolling average, window=5)'
        - 'XMEAS_2_lag_3' -> 'Reactor Temperature (lagged by 3 steps)'
        - 'XMEAS_3_diff' -> 'Reactor Pressure (first difference)'

    Note:
        - Assumes feature names follow underscore-separated naming convention
        - Base feature is extracted from first two underscore-separated parts
        - Supports rolling statistics (mean, min, max, std), lag features, and differences
        - Falls back to base description if no transformation is recognized
        - Rolling window size is extracted from the last part of the feature name
        - Lag step size is extracted from the last part of the feature name
    """
    parts = feature_name.split('_')

    base_feature = '_'.join(parts[:2])

    base_desc = descriptions[base_feature]

    if 'rolling' in feature_name:
        if 'mean' in feature_name:
            transform_desc = "rolling average"
        elif 'min' in feature_name:
            transform_desc = "rolling minimum"
        elif 'max' in feature_name:
            transform_desc = "rolling maximum"
        elif 'std' in feature_name:
            transform_desc = "rolling standard deviation"
        feature_name = f"{base_desc} ({transform_desc}, window={parts[-1]})"

    elif 'lag' in feature_name:
        feature_name = f"{base_desc} (lagged by {parts[-1]} steps)"

    elif 'diff' in feature_name:
        feature_name = f"{base_desc} (first difference)"

    else:
        feature_name = base_desc

    return feature_name
