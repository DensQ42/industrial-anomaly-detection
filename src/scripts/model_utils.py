import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
import optuna as opt
import time


class ZScoreDetector:
    """
    A Z-score based anomaly detector for time series and multivariate data.

    This class implements a statistical anomaly detection method using Z-scores
    (standard scores) to identify outliers in data. The detector calculates how
    many standard deviations each data point is from the mean, and flags points
    that exceed a specified threshold as anomalies. This approach is particularly
    effective for detecting point anomalies in data that follows approximately
    normal distributions.

    The Z-score method is computationally efficient and interpretable, making it
    suitable for real-time anomaly detection in industrial processes, sensor
    monitoring, and quality control applications. It works well when the normal
    operating conditions of the system can be characterized by stable mean and
    variance parameters.

    Attributes:
        threshold (float): The Z-score threshold above which points are considered
            anomalous. Common values are 2 (95% confidence) or 3 (99.7% confidence).
        mean_ (np.ndarray): The mean values for each feature, calculated during fitting.
        std_ (np.ndarray): The standard deviation values for each feature, calculated
            during fitting.
        is_fitted (bool): Flag indicating whether the detector has been fitted to data.
    """

    def __init__(self, threshold: float = 3) -> None:
        """
        Initializes the Z-score anomaly detector with specified threshold.

        Args:
            threshold (float, optional): The Z-score threshold for anomaly detection.
                Data points with absolute Z-scores exceeding this value are classified
                as anomalies. Higher values reduce false positives but may miss
                subtle anomalies. Defaults to 3.
        """
        self.threshold = threshold
        self.mean_ = None
        self.std_ = None
        self.is_fitted = False

    def fit(self, X_normal: np.ndarray):
        """
        Fits the detector using normal (non-anomalous) data to learn baseline statistics.

        This method calculates the mean and standard deviation for each feature
        using only normal operating data. These statistics define the baseline
        against which future data points will be compared for anomaly detection.
        It's crucial that the training data contains only normal observations
        to avoid biased baseline statistics.

        Args:
            X_normal (np.ndarray): Training data containing only normal observations.
                Shape should be (n_samples, n_features) for multivariate data or
                (n_samples,) for univariate data. Should not contain any known
                anomalies to ensure accurate baseline statistics.

        Returns:
            ZScoreDetector: Returns self to enable method chaining.

        Note:
            - Zero standard deviations are replaced with 1e-6 to prevent division by zero
            - Only normal (non-anomalous) data should be used for fitting
            - The detector assumes the normal data follows approximately Gaussian distribution
        """
        self.mean_ = np.mean(X_normal, axis=0)
        self.std_ = np.std(X_normal, axis=0)
        self.std_[self.std_ == 0] = 1e-6
        self.is_fitted = True
        return self

    def predict(self, X:np.ndarray) -> np.ndarray:
        """
        Predicts anomalies in the input data using fitted Z-score thresholds.

        This method calculates Z-scores for each data point relative to the baseline
        statistics learned during fitting. Points with absolute Z-scores exceeding
        the threshold in any feature are classified as anomalies. The method returns
        binary predictions where 1 indicates an anomaly and 0 indicates normal behavior.

        Args:
            X (np.ndarray): Input data for anomaly detection. Shape should match
                the training data format: (n_samples, n_features) for multivariate
                data or (n_samples,) for univariate data.

        Returns:
            np.ndarray: Binary array of shape (n_samples,) where 1 indicates
            anomaly and 0 indicates normal behavior.

        Raises:
            ValueError: If the detector has not been fitted to training data.

        Note:
            - Uses absolute Z-scores to detect deviations in either direction
            - For multivariate data, a sample is anomalous if ANY feature exceeds threshold
            - Z-score calculation: |X - mean| / std
        """
        if not self.is_fitted:
            raise ValueError('Model is not fitted')

        z_scores = np.abs((X - self.mean_) / self.std_)
        anomalies = np.any(z_scores > self.threshold, axis=1)
        return anomalies.astype(int)

    def set_params(self, *, threshold:int):
        """
        Sets the parameters of the detector.

        This method provides scikit-learn compatible parameter setting functionality,
        enabling the detector to be used with hyperparameter tuning tools like
        GridSearchCV and RandomizedSearchCV. The method allows dynamic adjustment
        of the threshold parameter without requiring model refitting.

        Args:
            threshold (float): The new Z-score threshold for anomaly detection.
                Must be a positive number. Higher values make detection more
                conservative (fewer false positives), while lower values make
                it more sensitive (fewer false negatives).

        Returns:
            ZScoreDetector: Returns self to enable method chaining.

        Note:
            - Uses keyword-only arguments to match scikit-learn conventions
            - Changing threshold does not require refitting the model
            - Compatible with scikit-learn parameter tuning workflows
        """
        self.threshold = threshold
        return self

    def get_params(self, deep: bool = True) -> dict:
        """
        Gets the parameters of the detector.

        This method provides scikit-learn compatible parameter retrieval functionality,
        enabling the detector to be used with hyperparameter tuning and model
        inspection tools. It returns a dictionary of all configurable parameters.

        Args:
            deep (bool): Whether to return parameters of sub-estimators. This
                parameter is included for scikit-learn compatibility but is not
                used since this detector has no sub-estimators.

        Returns:
            dict: Dictionary containing all detector parameters. Currently includes
            only 'threshold' parameter.

        Note:
            - Required for scikit-learn compatibility and hyperparameter tuning
            - The 'deep' parameter is unused but required for API consistency
            - Returns all parameters that can be set via set_params()
        """
        return {'threshold': self.threshold}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Computes anomaly probabilities based on Z-scores.

        This method calculates the probability of each sample being
        normal (class 0) or anomalous (class 1), similar to sklearn's
        predict_proba output.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Array of shape (n_samples, 2) where each row is
            [P(normal), P(anomaly)].
        """
        if not self.is_fitted:
            raise ValueError('Model is not fitted')

        z_scores = np.abs((X - self.mean_) / self.std_)
        max_z = np.max(z_scores, axis=1)

        probs_anomaly = 1 / (1 + np.exp(-(max_z - self.threshold)))
        probs_normal = 1 - probs_anomaly

        return np.vstack([probs_normal, probs_anomaly]).T

    @property
    def feature_importances_(self):
        """
        Same as feature_importances_ in XGBClassifier.
        """
        if self.std_ is None:
            raise ValueError("Model must be fitted before accessing feature_importances_")

        importances = 1 / self.std_
        importances = np.abs(importances) / np.sum(np.abs(importances))
        return importances

    def __repr__(self) -> str:
        return f"ZScoreDetector(threshold={self.threshold})"


def objective_function(
    trial: opt.trial.Trial,
    *,
    model: BaseEstimator,
    model_name: str,
    params: dict,
    X_train: np.ndarray,
    X_train_unsupervised: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray) -> float:
    """
    Objective function for Optuna hyperparameter optimization studies.

    This function serves as the optimization target for Optuna trials, performing
    hyperparameter tuning for machine learning models in anomaly detection tasks.
    It handles both supervised and unsupervised learning approaches by automatically
    selecting the appropriate training strategy based on the model type. The function
    evaluates model performance using F1-score, which is particularly suitable for
    imbalanced anomaly detection datasets.

    The function dynamically suggests hyperparameters based on the provided parameter
    configuration, trains the model with suggested parameters, and returns the
    performance metric that Optuna will optimize. It supports different parameter
    types including float, integer, and categorical parameters with appropriate
    sampling strategies.

    Args:
        trial (opt.trial.Trial): Optuna trial object used for suggesting hyperparameter
            values during the optimization process.
        model (BaseEstimator): The machine learning model instance to optimize.
            Should be a scikit-learn compatible estimator with fit() and predict()
            methods.
        model_name (str): Name identifier for the model type. Used to determine
            the appropriate training strategy (supervised vs unsupervised) and
            any model-specific post-processing requirements.
        params (dict): Hyperparameter search space configuration. Each key represents
            a parameter name, and values are dictionaries specifying the parameter
            type and bounds. Supported formats:
            - Float: {'type': 'float', 'low': min_val, 'high': max_val, 'log': bool}
            - Integer: {'type': 'int', 'low': min_val, 'high': max_val, 'step': step_size}
            - Categorical: {'type': 'categorical', 'choices': [option1, option2, ...]}
        X_train (np.ndarray): Training feature data for supervised learning approaches.
        X_train_unsupervised (np.ndarray): Training feature data for unsupervised
            learning approaches (typically containing only normal samples).
        y_train (np.ndarray): Training target labels for supervised learning approaches.
        X_test (np.ndarray): Test feature data for model evaluation.
        y_test (np.ndarray): Test target labels for performance evaluation.

    Returns:
        float: F1-score performance metric on the test set. Higher values indicate
        better model performance, making this suitable for Optuna maximization.

    Note:
        - Automatically sets random_state=42 for reproducible results if the model supports it
        - Handles unsupervised models (IsolationForest, ZScoreDetector) with special training logic
        - Converts IsolationForest outputs (-1 for anomalies) to standard binary format (1 for anomalies)
        - Uses model cloning to avoid modifying the original model instance
        - F1-score balances precision and recall, making it suitable for anomaly detection evaluation
    """
    model_clone = clone(model)

    suggested_params = {}
    for k, v in params.items():
        if v['type'] == 'float':
            suggested_params[k] = trial.suggest_float(k, v['low'], v['high'], log=v['log'])
        elif v['type'] == 'int':
            suggested_params[k] = trial.suggest_int(k, v['low'], v['high'], step=v['step'])
        elif v['type'] == 'categorical':
            suggested_params[k] = trial.suggest_categorical(k, v['choices'])

    if 'random_state' in model.get_params():
        suggested_params['random_state'] = 42

    model_clone.set_params(**suggested_params)

    unsupervised_models = ['Isolation Forest', 'Z-Score Detector']

    if model_name in unsupervised_models:
        model_clone.fit(X_train_unsupervised)
        y_pred = model_clone.predict(X_test)
    else:
        model_clone.fit(X_train, y_train)
        y_pred = model_clone.predict(X_test)

    if model_name == 'Isolation Forest':
        y_pred = (y_pred == -1).astype(int)

    score = f1_score(y_true=y_test, y_pred=y_pred)

    return score


def search_best_hyperparameters(
    models: dict[str, object],
    params_grid: dict[str, dict],
    trials: dict[str, int],
    X_train: np.ndarray,
    X_train_unsupervised: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray) -> tuple[dict[str, dict], dict[str, float]]:
    """
    Performs comprehensive hyperparameter optimization for multiple machine learning models using Optuna.

    This function automates the hyperparameter tuning process for a collection of machine
    learning models by running Optuna optimization studies for each model independently.
    It handles both supervised and unsupervised learning approaches, making it suitable
    for diverse anomaly detection scenarios. The function returns the optimal hyperparameters
    and corresponding performance scores for each model, enabling easy comparison and
    selection of the best-performing configuration.

    The optimization process uses the F1-score as the evaluation metric, which is
    particularly appropriate for imbalanced anomaly detection tasks where both precision
    and recall are important. Each model is optimized independently with its own
    parameter search space and trial budget.

    Args:
        models (dict[str, object]): Dictionary mapping model names to model instances.
            Each model should be a scikit-learn compatible estimator with fit()
            and predict() methods. Keys serve as identifiers for results reporting.
        params_grid (dict[str, dict]): Dictionary mapping model names to their respective
            hyperparameter search spaces. Each search space should follow the format
            expected by the objective_function, with parameter specifications including
            type, bounds, and sampling options.
        trials (dict[str, int]): Dictionary mapping model names to the number of
            optimization trials to run for each model. Higher values provide more
            thorough search but require more computation time.
        X_train (np.ndarray): Training feature data for supervised learning models.
        X_train_unsupervised (np.ndarray): Training feature data for unsupervised
            learning models (typically containing only normal samples for anomaly detection).
        y_train (np.ndarray): Training target labels for supervised learning models.
        X_test (np.ndarray): Test feature data used for model evaluation during optimization.
        y_test (np.ndarray): Test target labels used for performance evaluation.

    Returns:
        tuple[dict[str, dict], dict[str, float]]: A tuple containing two dictionaries:
            - best_hyperparams: Maps model names to their optimal hyperparameter sets
            - best_results: Maps model names to their best F1-scores achieved

    Note:
        - Each model gets an independent Optuna study for parallel optimization capability
        - Progress is printed for each model including best parameters and F1-score
        - Uses maximization direction since higher F1-scores indicate better performance
        - Includes small delay between models to prevent potential resource conflicts
        - Supports both supervised and unsupervised learning paradigms automatically
    """
    best_hyperparams = {}
    best_results = {}

    for model_name, model in models.items():

        study = opt.create_study(
            direction='maximize',
            study_name=model_name,
        )

        study.optimize(
            lambda trial: objective_function(
                trial=trial,
                model=model,
                model_name=model_name,
                params=params_grid[model_name],
                X_train=X_train,
                X_train_unsupervised=X_train_unsupervised,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            ),
            n_trials=trials[model_name],
        )

        best_hyperparams[model_name] = study.best_params
        best_results[model_name] = study.best_value

        print('Model name:', model_name)
        print('Best parameters:', study.best_params)
        print(f'Best F1:', round(study.best_value, 6), end='\n\n')

        time.sleep(1)

    return best_hyperparams, best_results


def prepare_models(models: dict[str, BaseEstimator],
                   hyperparams: dict[str, dict]) -> dict[str, BaseEstimator]:
    """
    Clones and configures machine learning models with their optimized hyperparameters.

    This function creates independent copies of machine learning models and applies
    the specified hyperparameters to each clone. Model cloning ensures that the
    original model instances remain unmodified, which is essential for reproducible
    experiments and when the same base models need to be reused with different
    configurations. This function is typically used after hyperparameter optimization
    to prepare models for final training and evaluation.

    The function validates that hyperparameters are provided for all models and
    applies them systematically, creating a ready-to-use collection of configured
    models. This approach supports clean separation between model definition,
    hyperparameter optimization, and final model preparation phases.

    Args:
        models (dict[str, BaseEstimator]): Dictionary mapping model names to model
            instances. Each model should be a scikit-learn compatible estimator
            with set_params() method for parameter configuration.
        hyperparams (dict[str, dict]): Dictionary mapping model names to their
            respective hyperparameter dictionaries. Each hyperparameter dictionary
            should contain parameter names as keys and their optimized values.
            These are typically the results from hyperparameter optimization.

    Returns:
        dict[str, BaseEstimator]: Dictionary mapping model names to configured
        model clones. Each model is an independent copy with applied hyperparameters,
        ready for training and evaluation.

    Raises:
        KeyError: If hyperparameters are not provided for any model in the models
        dictionary. This ensures that all models have their parameters properly
        configured before use.

    Note:
        - Uses sklearn.base.clone() to create independent model copies
        - Original model instances remain unmodified for reusability
        - All models must have corresponding hyperparameters to prevent configuration errors
        - Configured models are ready for fit() and predict() operations
        - Supports any scikit-learn compatible estimator with set_params() method
    """
    base_models = {}
    for model_name, model in models.items():
        if model_name not in hyperparams:
            raise KeyError(f"Hyperparameters for model '{model_name}' are not provided.")

        model_clone = clone(model)
        model_clone.set_params(**hyperparams[model_name])
        base_models[model_name] = model_clone

    return base_models


def fit_models(models: dict[str, BaseEstimator],
               X_train: np.ndarray,
               y_train: np.ndarray,
               X_train_unsupervised: np.ndarray) -> dict[str, BaseEstimator]:
    """
    Trains machine learning models using appropriate data based on their learning paradigm.

    This function handles the training process for a collection of machine learning models,
    automatically selecting the appropriate training data and approach based on the model
    type. It distinguishes between supervised models that require labeled data and
    unsupervised models that train only on feature data. This dual approach is particularly
    important in anomaly detection scenarios where some models learn from normal data
    only (unsupervised) while others can leverage both normal and anomalous examples
    (supervised).

    The function modifies the input models in-place by fitting them with the appropriate
    data, making them ready for prediction tasks. This approach ensures that each model
    is trained with the most suitable data representation for its underlying algorithm.

    Args:
        models (dict[str, BaseEstimator]): Dictionary mapping model names to configured
            model instances. Each model should be ready for training with hyperparameters
            already applied.
        X_train (np.ndarray): Training feature data for supervised learning models.
            Contains both normal and anomalous samples when available.
        y_train (np.ndarray): Training target labels for supervised learning models.
            Indicates whether each sample is normal (0) or anomalous (1).
        X_train_unsupervised (np.ndarray): Training feature data for unsupervised
            learning models. Typically contains only normal samples to establish
            baseline behavior patterns for anomaly detection.

    Returns:
        dict[str, BaseEstimator]: Dictionary mapping model names to fitted model instances.
        The same dictionary that was passed in, but with all models trained and ready
        for prediction.

    Note:
        - Unsupervised models (Z-Score Detector, Isolation Forest) use X_train_unsupervised
        - Supervised models use X_train and y_train for labeled training
        - Models are modified in-place and returned for convenience
        - Model identification relies on specific name matching for training strategy
        - All models are fitted and ready for predict() operations after this function
    """
    for name, model in models.items():
        if name in ['Z-Score Detector', 'Isolation Forest']:
            model.fit(X_train_unsupervised)
        else:
            model.fit(X_train, y_train)

    return models


def evaluate_models(models: dict[str, BaseEstimator],
                    X_test: np.ndarray,
                    y_test: np.ndarray) -> pd.DataFrame:
    """
    Evaluates classification models using comprehensive performance metrics.

    This function performs standardized evaluation of machine learning models for
    binary classification tasks, particularly focusing on anomaly detection scenarios.
    It calculates multiple performance metrics to provide a comprehensive view of
    model performance, including metrics that are robust to class imbalance which
    is common in anomaly detection problems.

    The function handles different model types by adapting the scoring methodology
    appropriately. For models that output probability scores, it uses predict_proba(),
    while for models like Isolation Forest that use different scoring mechanisms,
    it applies the appropriate transformations to ensure consistent evaluation.

    Args:
        models (dict[str, BaseEstimator]): Dictionary mapping model names to fitted
            model instances. Each model should be trained and ready for prediction.
        X_test (np.ndarray): Test feature data for model evaluation. Should have
            the same feature structure as the training data.
        y_test (np.ndarray): Test target labels for performance evaluation. Binary
            labels where 0 represents normal samples and 1 represents anomalies.

    Returns:
        pd.DataFrame: Evaluation results with models as rows and metrics as columns.
        Columns include:
            - F1: F1-score balancing precision and recall
            - Recall: True positive rate (sensitivity)
            - Precision: Positive predictive value
            - AUC: Area under the ROC curve

    Note:
        - Handles Isolation Forest's -1/1 output by converting to 0/1 format
        - Uses decision_function() for Isolation Forest scoring (negated for AUC)
        - Uses predict_proba()[:, 1] for other models' probability scores
        - All metrics are calculated using scikit-learn's standard implementations
        - Results are sorted by model names as they appear in the input dictionary
    """
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)

        if name == 'Isolation Forest':
            y_scores = -model.decision_function(X_test)
            y_pred = (y_pred == -1).astype(int)
        else:
            y_scores = model.predict_proba(X_test)[:, 1]

        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc_val = roc_auc_score(y_test, y_scores)

        results[name] = [f1, recall, precision, auc_val]

    df_results = pd.DataFrame.from_dict(results, orient='index',
                                        columns=['F1', 'Recall', 'Precision', 'AUC'])
    return df_results