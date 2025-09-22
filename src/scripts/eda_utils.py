import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.base import BaseEstimator
from sklearn.metrics import auc, roc_curve, confusion_matrix


def best_grid(n: int, max_cols: int = 5) -> tuple[int, int]:
    """
    Computes the optimal number of rows and columns to arrange `n` items in a grid.

    This function finds a visually balanced grid layout (rows × columns) for displaying
    `n` items, such as images or plots. It tries to find the smallest grid that fits all
    elements without exceeding `max_cols` columns. If an exact grid match is not found,
    it returns the smallest possible grid that fits all elements within the column limit.

    Args:
        n (int): Total number of items to arrange in a grid.
        max_cols (int, optional): Maximum number of columns allowed in the grid layout. Defaults to 5.

    Returns:
        tuple[int, int]: A tuple (rows, cols) representing the optimal number of rows and columns.
    """

    if n <= max_cols**2:
        for i in range(1, max_cols + 1):
            for j in range(1, max_cols + 1):
                if n == i * j:
                    return i, j
        for i in range(1, max_cols + 1):
            for j in range(1, max_cols + 1):
                if n <= i * j:
                    return i, j
    else:
        j = max_cols
        i = -(-n // j)
        return i, j


def histplot_plus_norm(data: pd.Series, title: str = '', bins: int = 30) -> None:
    """
    Plots a histogram with KDE and overlays a fitted normal distribution.

    This function visualizes the distribution of the input data using a histogram
    and kernel density estimation (KDE), and overlays the probability density
    function (PDF) of a normal distribution fitted to the data.

    The plot is styled to remove x and y axis ticks and gridlines for a cleaner visual.

    Args:
        data (pd.Series): The input data series to visualize.
        title (str, optional): The title of the plot. Defaults to an empty string.
        bins (int, optional): The number of bins to use for the histogram. Defaults to 30.

    Returns:
        None: This function only displays the plot and does not return any value.

    Example:
        >>> histplot_plus_norm(df['SalePrice'], title='Sale Price Distribution', bins=40)
    """
    mu, sigma = norm.fit(data)
    xx = np.linspace(min(data), max(data), 500)

    sns.histplot(data, kde=True, stat='density', bins=bins)
    plt.plot(xx, norm.pdf(xx, mu, sigma), 'r-', label='Normal Fit')

    plt.title(title)
    plt.gca().set_yticks([])
    plt.gca().set_xticks([])
    plt.grid(False)


def plot_distributions_with_transforms(data: pd.Series) -> None:
    """
    Plots the distribution of the original data and its transformations (square root and logarithmic).

    This function visualizes:
        1. The original distribution.
        2. The square root transformed distribution.
        3. The logarithmic transformed distribution.

    Each distribution is displayed with a histogram, KDE, and a fitted normal distribution
    using the `histplot_plus_norm` function.

    Args:
        data (pd.Series): The input data series to visualize.

    Returns:
        None: This function only displays plots and does not return any value.

    Example:
        >>> plot_distributions_with_transforms(df['SalePrice'])
    """
    plt.figure(figsize=(12, 3), dpi=200)

    plt.subplot(1, 3, 1)
    histplot_plus_norm(data=data, title='Original Distribution')

    plt.subplot(1, 3, 2)
    histplot_plus_norm(data=np.sqrt(data), title='Square Root Transform')

    plt.subplot(1, 3, 3)
    histplot_plus_norm(data=np.log(data), title='Log Transform')

    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(data: pd.DataFrame,
                            columns: list = None,
                            numeric_only: bool = True,
                            split_by: str = None,
                            prefix_title: str = '',
                            figsize: tuple[int, int] = (5, 4),
                            dpi: int = 150,
                            max_cols: int = 2,
                            cmap: str = None,
                            annot: bool = True,
                            fmt: str = '.1f',
                            cbar: bool = False,
                            annot_kws: dict = None,
                            full_range: bool = True) -> None:
    """
    Creates correlation matrix heatmaps for numerical variables with optional data splitting.

    This function generates one or multiple correlation matrix heatmaps using seaborn's
    heatmap visualization. It supports splitting data by categories to compare correlation
    patterns across different subgroups (e.g., train/test data, different time periods).
    The function allows customization of the display range, color scheme, and annotations.

    Args:
        data (pd.DataFrame): The input DataFrame containing the data for correlation analysis.
        columns (list, optional): List of column names to include in the correlation matrix.
            If None, uses all columns from the DataFrame. Defaults to None.
        numeric_only (bool, optional): Whether to include only numeric columns in correlation
            calculation. Defaults to True.
        split_by (str, optional): Column name to split data by for separate correlation
            matrices. Each unique value creates a separate subplot. Defaults to None.
        prefix_title (str, optional): Prefix text to add before category names in subplot
            titles when split_by is used. Defaults to ''.
        figsize (tuple[int, int], optional): Figure size as (width, height) in inches.
            Defaults to (5, 4).
        dpi (int, optional): Resolution of the figure in dots per inch. Defaults to 150.
        max_cols (int, optional): Maximum number of columns in the subplot grid when
            split_by is used. Defaults to 2.
        cmap (str, optional): Colormap for the heatmap. If None, uses 'coolwarm'.
            Defaults to None.
        annot (bool, optional): Whether to annotate cells with correlation values.
            Defaults to True.
        fmt (str, optional): String formatting code for annotations (e.g., '.2f' for
            two decimal places). Defaults to '.1f'.
        cbar (bool, optional): Whether to show the color bar legend. Defaults to False.
        annot_kws (dict, optional): Dictionary of keyword arguments for annotation text
            properties. If None, uses default font size of 8. Defaults to None.
        full_range (bool, optional): Whether to display full correlation range (-1 to 1).
            If False, shows absolute correlation values (0 to 1). Defaults to True.

    Returns:
        None: This function displays correlation heatmap(s) but does not return any value.

    Note:
        - When full_range=True: uses range [-1, 1] with center at 0
        - When full_range=False: uses absolute values [0, 1] without centering
        - Multiple subplots are arranged in a grid when split_by is specified
        - Subplot titles combine prefix_title with category values
        - Main title automatically adapts to single/multiple matrix display
    """
    if split_by is None:
        split_categories = [None]
    else:
        split_categories = data[split_by].unique().tolist()

    n = len(split_categories)
    h, w = best_grid(n, max_cols=max_cols)

    if columns is None:
        columns = data.columns.tolist()

    if annot_kws is None:
        annot_kws = {'size': 8}

    if full_range:
        vmin, vmax, center = -1, 1, 0
    else:
        vmin, vmax, center = 0, 1, None

    if cmap is None:
        cmap = 'coolwarm'

    plt.figure(figsize=figsize, dpi=dpi)

    for i, category in enumerate(split_categories):
        if category is None:
            subset_data = data[columns]
        else:
            subset_data = data[data[split_by] == category][columns]

        corr_matrix = subset_data.corr(numeric_only=numeric_only)

        if not full_range:
            corr_matrix = corr_matrix.abs()

        plt.subplot(h, w, i + 1)
        sns.heatmap(
            data=corr_matrix,
            cmap=cmap,
            annot=annot,
            fmt=fmt,
            annot_kws=annot_kws,
            vmax=vmax,
            vmin=vmin,
            center=center,
            cbar=cbar,
        )

        if category is not None:
            plt.title(prefix_title + ' ' + str(category))

    title = 'Correlation Matrices' if n > 1 else 'Correlation Matrix'
    plt.suptitle(title, fontsize=18)

    plt.tight_layout()
    plt.show()


def plot_variability_analysis(data: pd.DataFrame,
                              hue: str,
                              columns: list,
                              normalize: bool = True,
                              figsize: tuple[int, int] = (8, 16),
                              dpi: int = 150,
                              cbar: bool = False,
                              annot: bool = True,
                              cmap: str = 'viridis',
                              fmt: str = '.2f',
                              title: str = 'Normalized Coefficient of Variation'):
    """
    Creates a heatmap visualization of coefficient of variation across variables and categories.

    This function calculates and visualizes the coefficient of variation (CV) for each
    variable within different categories defined by the hue parameter. The coefficient
    of variation is a standardized measure of dispersion that allows comparison of
    variability across variables with different scales and units. This is particularly
    useful for analyzing process variability across different fault conditions in
    industrial datasets like Tennessee Eastman Process (TEP) data.

    The visualization helps identify which variables show the most variability under
    different operating conditions or fault scenarios, making it valuable for feature
    selection in fault detection and process monitoring applications.

    Args:
        data (pd.DataFrame): The input DataFrame containing the variables to analyze.
        hue (str): Column name to use for categorical grouping. Each unique value
            defines a separate category for variability analysis (e.g., fault types,
            operating modes).
        columns (list): List of column names to analyze for variability. These should
            be numerical columns where coefficient of variation is meaningful.
        normalize (bool, optional): Whether to normalize CV values by the maximum CV
            value for each fault category. This helps compare relative variability
            patterns. Defaults to True.
        figsize (tuple[int, int], optional): Figure size as (width, height) in inches.
            Default (8, 16) works well for many variables. Defaults to (8, 16).
        dpi (int, optional): Resolution of the figure in dots per inch. Higher values
            produce sharper images. Defaults to 150.
        cbar (bool, optional): Whether to display the color bar legend showing the
            scale of coefficient of variation values. Defaults to False.
        annot (bool, optional): Whether to annotate each cell with the numerical
            coefficient of variation value. Defaults to True.
        cmap (str, optional): Colormap for the heatmap visualization. 'viridis'
            provides good contrast and is colorblind-friendly. Defaults to 'viridis'.
        fmt (str, optional): String formatting code for cell annotations. Controls
            decimal precision display. Defaults to '.2f'.
        title (str, optional): Title for the heatmap plot. Should describe what
            is being visualized. Defaults to 'Normalized Coefficient of Variation'.

    Returns:
        None: This function displays a heatmap but does not return any value.

    Note:
        - Coefficient of variation is calculated as standard deviation divided by mean
        - Variables with zero mean are assigned CV = 0 to avoid division by zero
        - When normalize=True, CV values are scaled by maximum CV within each category
        - Higher CV values indicate greater relative variability within that category
        - The heatmap uses variables as rows and categories (hue values) as columns
    """
    variability_data = []


    if columns is None:
        columns = data.index.tolist()


    categories = data[hue].unique().tolist()


    for col in columns:
        for fault in categories:

            fault_data = data[data[hue] == fault][col]


            cv = (fault_data.std() / fault_data.mean()
                    if fault_data.mean() != 0 else 0)


            variability_data.append({
                'variable': col,
                'fault': fault,
                'cv': cv,
                'std': fault_data.std(),
                'range': fault_data.max() - fault_data.min()
            })


    var_df = pd.DataFrame(variability_data)
    pivot_cv = var_df.pivot(index='variable', columns='fault', values='cv')


    if normalize:
        pivot_cv = pivot_cv.div(pivot_cv.max(axis=0), axis=1)


    plt.figure(figsize=figsize, dpi=dpi)
    sns.heatmap(pivot_cv,
                annot=annot,
                cmap=cmap,
                fmt=fmt,
                cbar=cbar)
    plt.title(title)
    plt.show()


def plot_fault_comparison_timeseries(data: pd.DataFrame,
                                     columns: list,
                                     faults: list,
                                     max_sim: int = 3,
                                     xline: int = None,
                                     figsize: tuple[int, int] = (20, 5),
                                     dpi: int = 150,
                                     palette: str = 'deep') -> None:
    """
    Creates time series plots comparing multiple simulation runs across different fault scenarios with optional event markers.

    This function generates comprehensive time series visualizations for Tennessee Eastman
    Process (TEP) or similar simulation datasets, displaying how process variables evolve
    over time under different fault conditions. Each variable gets its own figure with
    subplots for each fault scenario, allowing direct comparison of temporal patterns.
    Multiple simulation runs are overlaid within each fault subplot to show variability
    and consistency of fault effects.

    The visualization includes an optional vertical line marker that can indicate important
    events such as fault introduction time, process changes, or other significant temporal
    landmarks. This feature is particularly valuable for understanding the timing of
    fault impacts and system responses in industrial process analysis.

    Args:
        data (pd.DataFrame): The input DataFrame containing time series simulation data.
            Must include columns: 'simulationRun', 'faultNumber', 'sample', and the
            variables specified in the columns parameter.
        columns (list): List of column names representing process variables to plot.
            Each variable generates a separate figure with fault comparisons.
        faults (list): List of fault numbers to compare. Each fault will be displayed
            in a separate subplot within each figure.
        max_sim (int, optional): Maximum number of simulation runs to include in the
            visualization. Limits overlaid lines to maintain clarity. Defaults to 3.
        xline (int, optional): X-coordinate for vertical reference line. Useful for
            marking fault introduction time or other significant events. If None,
            no vertical line is drawn. Defaults to None.
        figsize (tuple[int, int], optional): Figure size as (width, height) in inches.
            Should accommodate the number of fault subplots. Defaults to (20, 5).
        dpi (int, optional): Resolution of the figure in dots per inch. Higher values
            produce sharper images. Defaults to 150.
        palette (str, optional): Seaborn color palette for distinguishing simulation
            runs. 'deep' provides good contrast between multiple runs. Defaults to 'deep'.

    Returns:
        None: This function displays multiple time series plots but does not return
        any value.

    Note:
        - Creates one figure per variable with all fault comparisons as subplots
        - Subplots share y-axis scale for direct value comparison across faults
        - Legend is disabled to reduce visual clutter with multiple simulation runs
        - Vertical reference line appears on all subplots when xline is specified
        - Number of subplots determined by unique fault numbers in the dataset
        - Black dashed vertical line indicates the specified event marker
    """
    for col in columns:
        fig, axes = plt.subplots(1, len(data['faultNumber'].unique()), figsize=figsize, dpi=dpi, sharey=True)

        for i, fault in enumerate(faults):
            mask = (data['simulationRun'] <= max_sim) & (data['faultNumber'] == fault)
            df = data[mask]

            sns.lineplot(data=df,
                         x='sample',
                         y=col,
                         hue='simulationRun',
                         ax=axes[i],
                         palette=palette,
                         legend=False)

            axes[i].set_title(f'Fault {fault}')
            axes[i].set_xlabel('Time step')
            axes[i].set_ylabel('Value')

            if xline is not None:
                axes[i].axvline(x=xline, color='black', linestyle='--', linewidth=1.5)

        fig.suptitle(f'Feature: {col}', fontsize=16)

        plt.tight_layout()
        plt.show()


def plot_fault_comparison_boxplots(data: pd.DataFrame,
                                   columns: list,
                                   palette: str = 'deep',
                                   max_cols: int = 5,
                                   figsize: tuple[int, int] = (15, 15),
                                   dpi: int = 150) -> None:
    """
    Creates boxplot visualizations comparing variable distributions across different fault scenarios.

    This function generates a grid of boxplots to compare the statistical distributions
    of process variables under different fault conditions. Each subplot shows the
    distribution of one variable across all fault scenarios, making it easy to identify
    which faults cause significant changes in variable behavior. This visualization
    is particularly valuable for Tennessee Eastman Process (TEP) data analysis and
    fault detection research.

    Boxplots are effective for showing the central tendency, spread, and outliers
    in the data for each fault condition. The side-by-side comparison enables quick
    identification of fault-sensitive variables and understanding of how different
    faults affect process measurements.

    Args:
        data (pd.DataFrame): The input DataFrame containing simulation data with fault
            information. Must include a 'faultNumber' column and the variables
            specified in the columns parameter.
        columns (list): List of column names representing process variables to analyze.
            Each variable will be displayed in a separate subplot as a boxplot
            comparison across fault scenarios.
        palette (str, optional): Seaborn color palette for distinguishing different
            fault numbers. 'deep' provides good contrast between fault categories.
            Defaults to 'deep'.
        max_cols (int, optional): Maximum number of columns in the subplot grid layout.
            Used by best_grid() function to determine optimal arrangement. Defaults to 5.
        figsize (tuple[int, int], optional): Figure size as (width, height) in inches.
            Should be sized appropriately for the number of subplots. Defaults to (15, 15).
        dpi (int, optional): Resolution of the figure in dots per inch. Higher values
            produce sharper images but larger file sizes. Defaults to 150.

    Returns:
        None: This function displays boxplot comparisons but does not return any value.

    Note:
        - Each variable gets its own subplot showing fault number comparisons
        - Individual subplot legends are removed to reduce visual clutter
        - A single shared legend is placed at the bottom of the figure
        - Uses best_grid() function to determine optimal subplot arrangement
        - Boxplots show median, quartiles, and outliers for each fault scenario
        - Figure layout is automatically adjusted to accommodate the bottom legend
    """

    fig = plt.figure(figsize=figsize, dpi=dpi)

    n = len(columns)
    h, w = best_grid(n, max_cols=max_cols)

    axes = []

    for i, col in enumerate(columns):
        ax = plt.subplot(h, w, 1 + i)

        sns.boxplot(data=data,
                    y=col,
                    hue='faultNumber',
                    palette=palette,
                    ax=ax)

        axes.append(ax)

    handles, labels = axes[-1].get_legend_handles_labels()

    for ax in axes:
        ax.get_legend().remove()

    fig.legend(
        handles, labels,
        title='faultNumber',
        loc='lower center',
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(labels),
        frameon=False
    )

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.show()


def plot_mutual_info(data: pd.DataFrame,
                     target_column: str,
                     feature_columns: list = None,
                     figsize: tuple[int, int] = (10, 2),
                     dpi: int = 200,
                     rotation: int = 90) -> None:
    """
    Visualizes mutual information scores between features and a target variable.

    This function calculates and displays mutual information scores, which measure
    the amount of information one variable provides about another. Mutual information
    is a powerful feature selection metric that can capture both linear and non-linear
    relationships between variables, making it more comprehensive than correlation
    analysis. The visualization helps identify which features are most informative
    for predicting the target variable.

    The function automatically detects whether the target is categorical (classification)
    or continuous (regression) and applies the appropriate mutual information calculation.
    Features are ranked by their mutual information scores, with higher scores indicating
    stronger relationships with the target variable.

    Args:
        data (pd.DataFrame): The input DataFrame containing features and target variable.
            All feature columns should be numerical for mutual information calculation.
        target_column (str): Name of the target variable column. This can be either
            categorical (for classification) or continuous (for regression).
        feature_columns (list, optional): List of feature column names to analyze.
            If None, uses all columns except the target column. Defaults to None.
        figsize (tuple[int, int], optional): Figure size as (width, height) in inches.
            Default (10, 2) creates a wide, compact visualization suitable for many
            features. Defaults to (10, 2).
        dpi (int, optional): Resolution of the figure in dots per inch. Higher values
            produce sharper images. Defaults to 200.
        rotation (int, optional): Rotation angle for x-axis feature labels in degrees.
            90 degrees provides vertical labels that work well with many features.
            Defaults to 90.

    Returns:
        None: This function displays a bar plot but does not return any value.

    Note:
        - Uses mutual_info_classif for integer targets (classification)
        - Uses mutual_info_regression for non-integer targets (regression)
        - Features are automatically sorted by mutual information score (descending)
        - Higher bars indicate stronger information content about the target
        - Y-axis ticks are hidden to focus attention on relative bar heights
        - Each feature bar has a different color for easy identification
    """
    if feature_columns is None:
        X = data.loc[:, data.columns != target_column]
    else:
        X = data[feature_columns]

    y = data[target_column]

    if y.dtype == int:
        mi_scores = pd.Series(mutual_info_classif(X, y), index=X.columns)
    else:
        mi_scores = pd.Series(mutual_info_regression(X, y), index=X.columns)

    mi_scores = mi_scores.sort_values(ascending=False)

    plt.figure(figsize=figsize, dpi=dpi)

    sns.barplot(y=mi_scores,
                x=mi_scores.index,
                hue=mi_scores.index,
                palette='deep',
                dodge=False)

    plt.title(f'Mutual information for {target_column}')
    plt.xticks(rotation=rotation)
    plt.xlabel('Features')
    plt.ylabel('')

    plt.yticks([])

    plt.show()


def plot_histograms(data: pd.DataFrame,
                    columns: list = None,
                    bins: int = 25,
                    hue: str = None,
                    multiple: str = 'layer',
                    stat: str = 'probability',
                    palette: str = 'deep',
                    max_cols: int = 5,
                    figsize: tuple[int, int] = (14, 7),
                    dpi: int = 150) -> None:
    """
    Creates a grid of histogram plots for comparing distributions across variables and categories.

    This function generates multiple histogram visualizations arranged in a grid layout,
    allowing for comprehensive distribution analysis of multiple variables simultaneously.
    It supports categorical grouping through the hue parameter, making it valuable for
    comparing distributions between different groups or conditions. The function is
    particularly useful for exploratory data analysis, quality control, and understanding
    data characteristics across different scenarios.

    The histogram visualization helps identify distribution shapes, outliers, and
    differences between groups, making it essential for statistical analysis and
    feature understanding in machine learning workflows.

    Args:
        data (pd.DataFrame): The input DataFrame containing the variables to analyze.
        columns (list, optional): List of column names to create histograms for.
            If None, creates histograms for all columns in the DataFrame.
            Defaults to None.
        bins (int, optional): Number of bins for the histograms. Higher values
            provide more detailed distribution views but may introduce noise.
            Defaults to 25.
        hue (str, optional): Column name for categorical grouping. Different
            categories will be displayed with different colors, enabling
            distribution comparisons between groups. Defaults to None.
        multiple (str, optional): Method for displaying multiple distributions
            when hue is specified. Options include 'layer', 'dodge', 'stack',
            'fill'. 'layer' overlays distributions for easy comparison.
            Defaults to 'layer'.
        stat (str, optional): Statistic to compute for each bin. Options include
            'count', 'frequency', 'probability', 'density'. 'probability'
            normalizes to show proportions. Defaults to 'probability'.
        palette (str, optional): Seaborn color palette for distinguishing
            different hue categories. 'deep' provides good contrast between
            groups. Defaults to 'deep'.
        max_cols (int, optional): Maximum number of columns in the subplot grid.
            Used by best_grid() function to determine optimal layout. Defaults to 5.
        figsize (tuple[int, int], optional): Figure size as (width, height) in inches.
            Should be sized appropriately for the number of subplots. Defaults to (14, 7).
        dpi (int, optional): Resolution of the figure in dots per inch. Higher
            values produce sharper images. Defaults to 150.

    Returns:
        None: This function displays histogram plots but does not return any value.

    Note:
        - Only the first subplot shows individual legends to reduce visual clutter
        - A shared legend is placed at the bottom when hue grouping is used
        - Uses best_grid() function for optimal subplot arrangement
        - Figure layout accommodates the bottom legend with appropriate spacing
        - Probability statistic allows fair comparison across groups with different sizes
    """
    if columns is None:
        columns = data.columns.tolist()

    fig = plt.figure(figsize=figsize, dpi=dpi)
    n = len(columns)
    h, w = best_grid(n, max_cols=max_cols)
    axes = []

    for i, c in enumerate(columns):
        ax = plt.subplot(h, w, 1 + i)
        sns.histplot(
            data=data, x=c, bins=bins,
            hue=hue, multiple=multiple,
            stat=stat, palette=palette,
            ax=ax, legend=(i == 0),
        )
        axes.append(ax)

    legend = axes[0].get_legend()
    if legend:
        handles, labels = legend.legend_handles, [t.get_text() for t in legend.get_texts()]
        legend.remove()
        fig.legend(
            handles, labels,
            title=hue,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=len(labels),
            frameon=False
        )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.show()



def plot_sizes(dfs:list, names:list) -> None:

    rows = [df.shape[0] for df in dfs]

    _, ax = plt.subplots(figsize=(6, 3), dpi=150)
    bars = ax.bar(names, rows)

    ax.set_title('Number of Rows')
    ax.set_ylabel('Rows')

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, str(height),
                ha='center', va='bottom')

    plt.show()


def plot_roc_curves(models: dict[str, BaseEstimator],
                    X_test: np.ndarray,
                    y_test: np.ndarray,
                    figsize: tuple[int, int] = (6,5),
                    dpi: int = 150) -> None:
    """
    Creates ROC curve comparisons for multiple classification models.

    This function generates Receiver Operating Characteristic (ROC) curves for
    multiple machine learning models on the same plot, enabling direct visual
    comparison of their performance across different classification thresholds.
    ROC curves are particularly valuable for anomaly detection evaluation as they
    show the trade-off between true positive rate (sensitivity) and false positive
    rate (1-specificity) across all possible classification thresholds.

    The visualization includes the Area Under the Curve (AUC) values for each model
    in the legend, providing a quantitative summary of performance. A diagonal
    reference line represents random classifier performance, helping to contextualize
    model results. The function handles different model types by appropriately
    extracting prediction scores for ROC computation.

    Args:
        models (dict[str, BaseEstimator]): Dictionary mapping model names to fitted
            model instances. Each model should be trained and capable of producing
            prediction scores for ROC analysis.
        X_test (np.ndarray): Test feature data for generating predictions. Should
            have the same feature structure as training data.
        y_test (np.ndarray): Test target labels for ROC computation. Binary labels
            where 0 represents normal samples and 1 represents anomalies.
        figsize (tuple[int, int], optional): Figure size as (width, height) in inches.
            Defaults to (10, 8).
        dpi (int, optional): Resolution of the figure in dots per inch. Higher values
            produce sharper images. Defaults to 150.

    Returns:
        None: This function displays the ROC curves plot but does not return any value.

    Note:
        - Uses decision_function() for Isolation Forest (negated for proper orientation)
        - Uses predict_proba()[:, 1] for models with probability outputs
        - Includes random classifier baseline for performance context
        - AUC values are displayed in legend with 3 decimal precision
        - Grid is enabled for easier curve reading
        - Axes are properly labeled with statistical terminology
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    for name, model in models.items():
        if name == 'Isolation Forest':
            y_scores = -model.decision_function(X_test)
        elif name == 'Z-Score Detector':
            y_scores = model.predict_proba(X_test)[:, 1]
        else:
            y_scores = model.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_scores)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--',
            label='Random Classifier (AUC = 0.500)')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    ax.set_title('ROC Curves Comparison',
                    fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11, frameon=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(model: BaseEstimator,
                          X: np.ndarray,
                          y: np.ndarray,
                          title: str = 'Confusion matrix',
                          *,
                          annot: bool = True,
                          annot_size: int = 16,
                          fmt: str = '.2f',
                          cbar: bool = False,
                          cmap: str = 'Blues') -> None:
    """
    Creates a confusion matrix heatmap visualization for binary classification models.

    This function generates a visual representation of classification performance by
    displaying the confusion matrix as a heatmap. The confusion matrix shows the
    relationship between actual and predicted classifications, making it easy to
    identify true positives, true negatives, false positives, and false negatives.
    This visualization is particularly valuable for anomaly detection tasks where
    understanding the types of classification errors is crucial for model assessment.

    The heatmap format provides an intuitive way to quickly assess model performance,
    with darker colors typically indicating higher values. The function is specifically
    designed for binary classification with 'Normal' and 'Faulty' class labels,
    making it suitable for industrial process monitoring and fault detection applications.

    Args:
        model (BaseEstimator): A fitted classification model with a predict() method.
            Should be trained and ready for making predictions on the provided data.
        X (np.ndarray): Feature data for generating predictions. Should have the same
            structure as the data used for model training.
        y (np.ndarray): True target labels for comparison with predictions. Binary
            labels where 0 represents normal samples and 1 represents faulty samples.
        annot (bool, optional): Whether to annotate cells with numerical values.
            When True, displays the count of samples in each cell. Defaults to True.
        annot_size (int, optional): Font size for cell annotations. Larger values
            improve readability but may not fit in smaller matrices. Defaults to 16.
        fmt (str, optional): String formatting for cell annotations. Use 'd' for
            integers or '.2f' for floating point display. Defaults to '.2f'.
        cbar (bool, optional): Whether to display the color scale bar. Generally
            not needed for confusion matrices as values are self-explanatory.
            Defaults to False.
        cmap (str, optional): Colormap for the heatmap. 'Blues' provides good
            contrast and readability for confusion matrices. Defaults to 'Blues'.

    Returns:
        None: This function displays the confusion matrix heatmap but does not
        return any value.

    Note:
        - Automatically generates predictions using model.predict()
        - Labels are hardcoded as 'Normal' and 'Faulty' for binary classification
        - Matrix layout: rows represent true labels, columns represent predictions
        - Top-left cell shows True Negatives, bottom-right shows True Positives
        - Uses seaborn heatmap for consistent styling and formatting
    """
    y_pred = model.predict(X)

    cm = confusion_matrix(y, y_pred)
    sns.heatmap(data=cm,
                annot=annot,
                fmt=fmt,
                cmap=cmap,
                cbar=cbar,
                annot_kws={'size': annot_size},
                xticklabels=['Normal', 'Faulty'],
                yticklabels=['Normal', 'Faulty'])

    plt.title(title, fontsize=18)
    plt.ylabel('True labels', fontsize=14)
    plt.xlabel('Predicted labels', fontsize=14)


def plot_feature_importance(models: dict[str, BaseEstimator],
                            model_names: list[str],
                            columns: list[str],
                            max_features: int = 10,
                            max_cols: int = 3,
                            figsize: tuple[int, int] = (12, 4),
                            dpi: int = 150,
                            palette: str = 'deep') -> None:
    """
    Visualizes feature importance rankings for tree-based machine learning models.

    This function creates a grid of horizontal bar plots displaying the most important
    features for each specified model. Feature importance values indicate how much
    each feature contributes to the model's decision-making process, with higher
    values representing more influential features. This visualization is essential
    for understanding model behavior, feature selection, and identifying the most
    critical variables for prediction tasks.

    The function displays features in descending order of importance, making it easy
    to identify the top contributors to model performance. This analysis is particularly
    valuable for anomaly detection and fault diagnosis applications where understanding
    which process variables are most indicative of different conditions is crucial
    for both model interpretation and domain insights.

    Args:
        models (dict[str, BaseEstimator]): Dictionary mapping model names to fitted
            model instances. Models must have feature_importances_ attribute
            (typically tree-based models like Random Forest, XGBoost, etc.).
        model_names (list[str]): List of model names to include in the visualization.
            These names should correspond to keys in the models dictionary.
        columns (list[str]): List of feature names corresponding to the columns
            used during model training. Should match the order of features used
            in the training data.
        max_features (int, optional): Maximum number of top features to display
            for each model. Limits the visualization to the most important features.
            Defaults to 10.
        max_cols (int, optional): Maximum number of columns in the subplot grid.
            Used by best_grid() function to determine optimal layout. Defaults to 3.
        figsize (tuple[int, int], optional): Figure size as (width, height) in inches.
            Should accommodate the number of model subplots. Defaults to (12, 4).
        dpi (int, optional): Resolution of the figure in dots per inch. Higher values
            produce sharper images. Defaults to 150.
        palette (str, optional): Seaborn color palette for the bar plots. Different
            colors help distinguish features within each subplot. Defaults to 'deep'.

    Returns:
        None: This function displays feature importance plots but does not return
        any value.

    Note:
        - Only works with models that have feature_importances_ attribute
        - Features are automatically sorted by importance in descending order
        - Each model gets its own subplot with horizontal bar orientation
        - Legend is disabled to reduce visual clutter since colors are for aesthetics
        - Y-axis labels are removed from subplot titles for cleaner appearance
    """
    plt.figure(figsize=figsize, dpi=dpi)
    h, w = best_grid(len(model_names), max_cols=max_cols)

    for i, model_name in enumerate(model_names):
        importance_df = pd.DataFrame({'Feature': columns})
        importance_df['Importance'] = models[model_name].feature_importances_
        importance_df = importance_df.sort_values('Importance', ascending=False)
        top = importance_df.head(max_features)

        plt.subplot(h, w, 1 + i)
        sns.barplot(data=top, y='Feature', x='Importance', hue='Feature',
                    palette=palette, legend=False)
        plt.title(model_name)
        plt.ylabel('')

    plt.tight_layout()
    plt.show()



def plot_predictions_on_timeseries(df_test: pd.DataFrame,
                                   model: BaseEstimator,
                                   scaler,
                                   feature_columns: list,
                                   fault_type: int,
                                   simulation_id: int,
                                   key_variables: list,
                                   model_name: str) -> np.ndarray:
    """
    Visualizes anomaly detection predictions overlaid on time series data for specific fault scenarios.

    This function creates comprehensive time series visualizations that show both the
    original variable behavior and the model's anomaly predictions for a specific
    fault scenario and simulation run. It provides visual validation of model performance
    by overlaying predicted anomalies as red scatter points on the time series plots.
    For fault conditions, it also highlights the true anomaly period to enable direct
    comparison between predicted and actual anomalous regions.

    The visualization is particularly valuable for understanding temporal patterns of
    anomaly detection, assessing model sensitivity, and identifying whether the model
    correctly captures the onset and duration of fault conditions. Each key variable
    gets its own subplot to show how different process variables respond to faults
    and how well the model detects these changes.

    Args:
        df_test (pd.DataFrame): Test dataset containing time series data with columns
            for 'faultNumber', 'simulationRun', 'sample', 'faulty', and process variables.
        model (BaseEstimator): Fitted anomaly detection model with predict() method.
            Should be trained and ready for making predictions.
        scaler: Fitted data scaler (e.g., StandardScaler, MinMaxScaler) used to
            transform features to the same scale as during model training.
        feature_columns (list): List of column names used as input features for
            the model. Should match the features used during training.
        fault_type (int): Fault number to analyze. Use 0 for normal operation
            or specific fault numbers for faulty conditions.
        simulation_id (int): Simulation run identifier to focus the analysis
            on a specific experimental run.
        key_variables (list): List of important process variables to visualize.
            These are the variables that will be plotted with anomaly overlays.
        model_name (str): Name of the model for plot title and identification.

    Returns:
        np.ndarray: Array of anomaly predictions (0 for normal, 1 for anomaly)
        for the specified fault and simulation scenario.

    Note:
        - Red scatter points indicate predicted anomalies
        - Orange shaded region shows true anomaly period for fault conditions
        - Normal operation (fault_type=0) shows no true anomaly highlighting
        - Each subplot corresponds to one key variable
        - X-axis represents sample/time points, Y-axis shows variable values
        - Model predictions are made on scaled data but plotted on original scale
    """
    data = df_test[(df_test['faultNumber'] == fault_type) &
                   (df_test['simulationRun'] == simulation_id)]
    X = data[feature_columns]

    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)

    fig, axes = plt.subplots(len(key_variables), 1,
                             figsize=(15, 3 * len(key_variables)))

    for i, var in enumerate(key_variables):
        axes[i].plot(data['sample'], data[var], label=None, alpha=0.7)

        anomaly_mask = predictions == 1
        if anomaly_mask.any():
            axes[i].scatter(data['sample'][anomaly_mask],
                          data[var][anomaly_mask],
                          color='red', s=20, label='Predicted Anomaly')

        if fault_type != 0:
            true_anomaly_mask = data['faulty'] == True
            axes[i].axvspan(data[true_anomaly_mask]['sample'].min() if true_anomaly_mask.any() else 0,
                        data['sample'].max(), alpha=0.2, color='orange',
                        label='True Anomaly Period')

        axes[i].set_title(f"Fault type {fault_type}  -  Feature {var}")
        axes[i].legend()

    plt.suptitle(f'Model {model_name}', fontsize=16)
    plt.tight_layout()
    plt.show()

    return predictions