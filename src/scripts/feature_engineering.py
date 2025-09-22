import pandas as pd

def create_lag_features(data: pd.DataFrame,
                        lags: list,
                        columns: list = None,
                        group_by: str = None,
                        dropna: bool = True) -> pd.DataFrame:
    """
    Creates lagged versions of specified columns with optional grouping functionality.

    This function generates time-lagged features by shifting specified columns by
    different time steps, with the ability to perform lagging within specific groups.
    The grouping functionality is particularly useful for datasets containing multiple
    time series (e.g., multiple sensors, different fault scenarios, or separate
    simulation runs) where lags should be calculated independently within each group
    rather than across group boundaries.

    When group_by is specified, the function ensures that lag features do not
    inappropriately mix data across different logical sequences, maintaining the
    integrity of temporal relationships within each group.

    Args:
        data (pd.DataFrame): The input DataFrame containing time series data.
            Should be ordered chronologically, especially within groups if grouping
            is used.
        lags (list): List of integers specifying the lag periods to create.
            Positive values create backward lags (previous time steps).
            For example, [1, 2, 5] creates lags of 1, 2, and 5 time steps.
        columns (list, optional): List of column names to create lag features for.
            If None, creates lag features for all columns in the DataFrame.
            Defaults to None.
        group_by (str, optional): Column name to group by before creating lag features.
            When specified, lags are calculated independently within each group,
            preventing data leakage across different sequences. Common use cases
            include grouping by simulation runs, fault numbers, or sensor IDs.
            Defaults to None.
        dropna (bool, optional): Whether to remove rows with NaN values that result
            from lagging. When True, removes rows with any NaN values. When False,
            keeps all rows including those with NaN from lagging operations.
            Defaults to True.

    Returns:
        pd.DataFrame: A new DataFrame containing the original data plus all lag
        features. Lag columns are named using the pattern '{column}_lag_{lag_value}'.
        If group_by is used, maintains the original order but with group-wise
        lag calculations. If dropna=True, returns DataFrame with NaN rows removed.

    Note:
        - When group_by is specified, lags are calculated independently within each group
        - Prevents temporal leakage across different logical sequences
        - Original DataFrame is copied to avoid modifying the input data
        - Column naming follows the pattern: original_name_lag_N
        - Useful for multi-sensor data, multiple simulation runs, or stratified analysis
        - Group-wise operation resets index to maintain clean sequential numbering
    """
    if columns is None:
        columns = list(data.columns)

    def create_lags_for_group(group):
        lagged_features = []

        for col in columns:
            for lag in lags:
                lagged_features.append(
                    group[col].shift(lag).rename(f"{col}_lag_{lag}")
                )

        lagged_df = pd.concat(lagged_features, axis=1)
        result = pd.concat([group.copy(), lagged_df], axis=1)

        if dropna:
            return result.dropna()
        else:
            return result

    if group_by is None:
        return create_lags_for_group(data)
    else:
        return data.groupby(group_by).apply(create_lags_for_group).reset_index(drop=True)


def create_rolling_features(data: pd.DataFrame,
                            window_sizes: list,
                            columns: list = None,
                            group_by: str = None,
                            dropna: bool = True) -> pd.DataFrame:
    """
    Creates rolling window statistical features with optional group-wise calculations.

    This function generates multiple rolling statistics (mean, standard deviation, minimum,
    and maximum) across different window sizes for time series data, with the ability to
    perform rolling calculations independently within specified groups. The grouping
    functionality is essential for datasets containing multiple time series sequences
    where rolling statistics should not cross group boundaries, ensuring temporal
    integrity within each logical sequence.

    Group-wise rolling features are particularly valuable for multi-sensor data,
    multiple simulation runs, or stratified time series analysis where each group
    represents a distinct temporal sequence that should be analyzed independently.

    Args:
        data (pd.DataFrame): The input DataFrame containing time series data.
            Should be ordered chronologically, especially within groups if grouping
            is used.
        window_sizes (list): List of integers specifying the rolling window sizes.
            Each window size will generate four statistical features (mean, std, min, max)
            for each specified column. For example, [5, 10, 20] creates features
            for 5-period, 10-period, and 20-period windows.
        columns (list, optional): List of column names to create rolling features for.
            If None, creates rolling features for all columns in the DataFrame.
            Defaults to None.
        group_by (str, optional): Column name to group by before creating rolling features.
            When specified, rolling statistics are calculated independently within each
            group, preventing data contamination across different sequences. Common
            use cases include grouping by simulation runs, fault numbers, or sensor IDs.
            Defaults to None.
        dropna (bool, optional): Whether to remove rows with NaN values that may
            result from rolling calculations. When True, removes rows with any NaN
            values. When False, keeps all rows including those with potential NaN values.
            Defaults to True.

    Returns:
        pd.DataFrame: A new DataFrame containing the original data plus all rolling
        statistical features. Rolling columns are named using the pattern
        '{column}_rolling_{statistic}_{window_size}'. If group_by is used, maintains
        group-wise calculations while resetting index for clean sequential numbering.

    Note:
        - Uses min_periods=1 to ensure calculations start from the first available value
        - Four statistics calculated for each column-window combination within groups
        - When group_by is specified, rolling windows respect group boundaries
        - Column naming pattern: original_name_rolling_statistic_windowsize
        - Prevents temporal contamination across different logical sequences
        - Group-wise operation resets index to maintain clean sequential numbering
        - Useful for multi-sequence datasets and stratified temporal analysis
    """
    if columns is None:
        columns = list(data.columns)

    def create_rolling_for_group(group):
        features = []
        for col in columns:
            for w in window_sizes:
                features.append(group[col].rolling(window=w, min_periods=1).mean().rename(f"{col}_rolling_mean_{w}"))
                features.append(group[col].rolling(window=w, min_periods=1).std().rename(f"{col}_rolling_std_{w}"))
                features.append(group[col].rolling(window=w, min_periods=1).min().rename(f"{col}_rolling_min_{w}"))
                features.append(group[col].rolling(window=w, min_periods=1).max().rename(f"{col}_rolling_max_{w}"))

        features_df = pd.concat(features, axis=1)
        result = pd.concat([group.copy(), features_df], axis=1)

        if dropna:
            return result.dropna()
        else:
            return result

    if group_by is None:
        return create_rolling_for_group(data)
    else:
        return data.groupby(group_by).apply(create_rolling_for_group).reset_index(drop=True)


def create_diff_features(data: pd.DataFrame,
                         columns: list = None,
                         group_by: str = None,
                         dropna: bool = True) -> pd.DataFrame:
    """
    Creates first-order difference features with optional group-wise calculations.

    This function calculates the first difference (current value minus previous value)
    for specified columns, with the ability to perform differencing independently
    within specified groups. The grouping functionality ensures that difference
    calculations do not cross group boundaries, preventing inappropriate temporal
    relationships between different logical sequences.

    Group-wise differencing is essential for datasets containing multiple time series
    where each group represents a distinct temporal sequence (e.g., different simulation
    runs, sensor readings, or experimental conditions) that should be analyzed
    independently to maintain temporal integrity.

    Args:
        data (pd.DataFrame): The input DataFrame containing time series data.
            Should be ordered chronologically, especially within groups if grouping
            is used.
        columns (list, optional): List of column names to create difference features for.
            If None, creates difference features for all columns in the DataFrame.
            Defaults to None.
        group_by (str, optional): Column name to group by before creating difference
            features. When specified, differences are calculated independently within
            each group, preventing temporal contamination across different sequences.
            Common use cases include grouping by simulation runs, fault numbers,
            or sensor IDs. Defaults to None.
        dropna (bool, optional): Whether to remove rows with NaN values that result
            from differencing operations. When True, removes rows with any NaN values.
            When False, keeps all rows including the first row of each group which
            will contain NaN for difference features. Defaults to True.

    Returns:
        pd.DataFrame: A new DataFrame containing the original data plus difference
        features. Difference columns are named using the pattern '{column}_diff'.
        If group_by is used, maintains group-wise calculations while resetting
        index for clean sequential numbering.

    Note:
        - First difference = current_value - previous_value
        - When group_by is specified, differences respect group boundaries
        - The first row of each group will contain NaN values for difference features
        - Original DataFrame is copied to avoid modifying the input data
        - Column naming follows the pattern: original_name_diff
        - Prevents temporal leakage across different logical sequences
        - Group-wise operation resets index to maintain clean sequential numbering
        - Useful for trend analysis and stationarity within independent sequences
    """
    if columns is None:
        columns = [col for col in data.columns]

    def create_diff_for_group(group):
        new_data = group.copy()

        for col in columns:
            new_data[f'{col}_diff'] = group[col].diff()

        if dropna:
            return new_data.dropna()
        else:
            return new_data

    if group_by is None:
        return create_diff_for_group(data)
    else:
        return data.groupby(group_by).apply(create_diff_for_group).reset_index(drop=True)
