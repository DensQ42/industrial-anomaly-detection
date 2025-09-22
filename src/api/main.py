import uvicorn, nest_asyncio, logging, warnings
import pandas as pd
from datetime import datetime
from io import StringIO
from fastapi import HTTPException, UploadFile, File, Form, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import List, Tuple, Dict, Any, Optional, Union
from src.api.models import CSVValidationResponse, HealthResponse, ModelInfo, AnalysisResponse
from src.pipeline.anomaly_detection_pipeline import ImprovedAnomalyDetectionPipeline
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

logging.basicConfig(level=logging.INFO)
api_logger = logging.getLogger('anomaly_api')


def validate_csv_data(df: pd.DataFrame) -> Tuple[bool, List[str], List[str]]:
    """
    Validates CSV data format and content for Tennessee Eastman Process anomaly detection.

    This function performs comprehensive validation of uploaded CSV data to ensure
    it meets the requirements for anomaly detection analysis. It checks for required
    columns, data types, data quality, and structural integrity. The validation
    covers both critical errors that prevent processing and warnings about potential
    issues that may affect analysis quality.

    The function is specifically designed for Tennessee Eastman Process (TEP) datasets,
    which include standardized measurement variables (xmeas_1 through xmeas_41) and
    manipulated variables (xmv_1 through xmv_11), along with temporal and experimental
    metadata.

    Args:
        df (pd.DataFrame): The uploaded CSV data as a pandas DataFrame to validate
            against TEP dataset requirements and anomaly detection processing needs.

    Returns:
        Tuple[bool, List[str], List[str]]: A tuple containing:
            - is_valid (bool): Whether the data passes all critical validation checks
            - errors (List[str]): List of critical errors that prevent processing
            - warnings (List[str]): List of non-critical issues that may affect quality

    Note:
        - Requires minimum 3 rows for meaningful temporal analysis
        - Expects 41 measurement variables (xmeas_1 to xmeas_41)
        - Expects 11 manipulated variables (xmv_1 to xmv_11)
        - Critical columns (sample, simulationRun) cannot have missing values
        - Warns about non-monotonic sample ordering and multiple simulation runs
        - All feature columns must contain numeric data types
    """
    errors = []
    warns = []

    if len(df) < 3:
        errors.append('CSV must contain at least 3 rows for temporal analysis')
        return False, errors, warns

    required_columns = ['sample', 'simulationRun']
    missing_required = [col for col in required_columns if col not in df.columns]
    if missing_required:
        errors.append(f'Missing required columns: {missing_required}')

    required_features = ['xmeas_' + str(i) for i in range(1, 42, 1)] + ['xmv_' + str(i) for i in range(1, 12, 1)]
    missing_required = [col for col in required_features if col not in df.columns]
    if missing_required:
        errors.append(f'Missing required features: {missing_required}')

    non_numeric_cols = []
    for col in required_features:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            non_numeric_cols.append(col)

    if non_numeric_cols:
        errors.append(f'Non-numeric columns found: {non_numeric_cols[:3]}...')

    critical_missing = []
    for col in ['sample', 'simulationRun']:
        if col in df.columns and df[col].isnull().any():
            critical_missing.append(col)

    if critical_missing:
        errors.append(f'Missing values in critical columns: {critical_missing}')

    if 'sample' in df.columns:
        if not df['sample'].is_monotonic_increasing:
            warns.append('Sample values are not in ascending order')

    if 'simulationRun' in df.columns:
        unique_runs = df['simulationRun'].nunique()
        if unique_runs > 1:
            warns.append(f'Multiple simulation runs detected ({unique_runs}). Using the first one.')

    is_valid = len(errors) == 0
    return is_valid, errors, warns


def format_shap_features(impactful_features: List[tuple]) -> List[Dict[str, Any]]:
    """
    Converts SHAP feature importance data into a standardized API response format.

    This function transforms raw SHAP (SHapley Additive exPlanations) feature
    importance results into a consistent, API-friendly format suitable for client
    consumption. It extracts key information from SHAP analysis including feature
    names, human-readable descriptions, and quantitative importance scores,
    presenting them in a structured format that can be easily consumed by
    frontend applications or other API clients.

    The formatting ensures numerical precision control and consistent field naming
    across the API response, making it easier for clients to parse and display
    feature importance information in user interfaces or reports.

    Args:
        impactful_features (List[Tuple]): List of tuples where each tuple contains:
            - feature_name (str): Technical name of the feature
            - feature_data (dict): Dictionary containing feature information with keys:
                - 'description': Human-readable description of the feature
                - 'total_impact': Numerical importance score from SHAP analysis

    Returns:
        List[Dict[str, Any]]: List of dictionaries with standardized structure:
            - 'variable_name': Technical feature name
            - 'description': Human-readable feature description
            - 'importance_score': Rounded importance score (4 decimal places)

    Note:
        - Importance scores are rounded to 4 decimal places for consistency
        - Maintains original feature ordering from input list
        - Assumes input feature_data contains 'description' and 'total_impact' keys
        - Designed for integration with AnalysisResponse.important_features field
    """
    formatted_features = []
    for feature_name, feature_data in impactful_features:
        formatted_features.append({
            'variable_name': feature_name,
            'description': feature_data['description'],
            'importance_score': round(float(feature_data['total_impact']), 4)
        })
    return formatted_features


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """
    Manages the application lifecycle for FastAPI startup and shutdown operations.

    This async context manager handles the initialization and cleanup of critical
    application resources, specifically the anomaly detection pipeline. It ensures
    that expensive initialization operations (model loading, LLM service setup)
    occur once during application startup rather than on each request, and provides
    graceful cleanup during application shutdown.

    The lifespan manager follows FastAPI's recommended pattern for managing
    application state and resources that need to persist across the application
    lifetime. It provides proper error handling during initialization and ensures
    clean resource disposal during shutdown.

    Args:
        _app (FastAPI): The FastAPI application instance (parameter name prefixed
            with underscore to indicate it's not used in the function body).

    Yields:
        None: Control is yielded to FastAPI for normal application operation
        between startup and shutdown phases.

    Raises:
        Exception: Re-raises any initialization errors to prevent application
        startup with improperly configured resources.

    Note:
        - Global pipeline variable is set during startup for access across endpoints
        - Initialization errors are logged and re-raised to fail fast
        - Pipeline is explicitly set to None during shutdown for clean disposal
        - All logging uses the api_logger for consistent log formatting
        - Critical for ensuring model availability before serving requests
    """
    global pipeline
    try:
        api_logger.info('Initializing Anomaly Detection Pipeline...')
        pipeline = ImprovedAnomalyDetectionPipeline()
        api_logger.info('Pipeline initialized successfully')
    except Exception as e:
        api_logger.error(f'Failed to initialize pipeline: {str(e)}')
        raise e

    yield

    api_logger.info('Shutting down pipeline...')
    pipeline = None

pipeline = None

app = FastAPI(
    title='Tennessee Eastman Process Anomaly Detection API',
    description='ML-powered system for industrial process anomaly detection and LLM explanations',
    version='1.0.0',
    docs_url='/docs',
    redoc_url='/redoc',
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['GET', 'POST'],
    allow_headers=['*'],
)


@app.post('/validate-csv', response_model=CSVValidationResponse)
async def validate_csv_file(file: UploadFile = File(...)):
    """
    Validates uploaded CSV files for Tennessee Eastman Process anomaly detection compatibility.

    This endpoint performs comprehensive validation of CSV files to ensure they meet
    the format and content requirements for anomaly detection analysis. It checks
    file format, required columns, data types, and structural integrity while
    providing detailed feedback about any issues found. The validation helps users
    identify and correct problems before attempting analysis.

    The endpoint handles file upload, parsing, and validation in a single operation,
    returning structured feedback that distinguishes between critical errors and
    warnings. It also extracts metadata about the dataset structure for user
    reference and analysis planning.

    Args:
        file (UploadFile): The CSV file to validate, uploaded via multipart form data.
            Must have .csv extension and contain TEP-compatible data structure.

    Returns:
        CSVValidationResponse: Validation results including:
            - Overall validity status
            - List of errors and warnings
            - Dataset metadata (shape, simulation runs, sample range)

    Raises:
        HTTPException: May raise validation errors if file cannot be processed,
        though most errors are captured and returned in the response structure.

    Note:
        - Accepts only files with .csv extension
        - Assumes UTF-8 encoding for file contents
        - Validates against Tennessee Eastman Process dataset requirements
        - Returns detailed metadata for valid files
        - Gracefully handles parsing errors and malformed data
    """
    try:
        if not file.filename.lower().endswith('.csv'):
            return CSVValidationResponse(
                is_valid=False,
                errors=['File must be a CSV file'],
                warnings=[],
            )

        contents = await file.read()
        csv_string = contents.decode('utf-8')
        df = pd.read_csv(StringIO(csv_string))

        is_valid, errors, warns = validate_csv_data(df)

        if 'simulationRun' in df.columns:
            simulation_runs = df['simulationRun'].unique().tolist()
        else:
            simulation_runs = []

        if 'sample' in df.columns:
            sample_range = {'min': int(df['sample'].min()), 'max': int(df['sample'].max())}
        else:
            sample_range = None

        return CSVValidationResponse(
            is_valid=is_valid,
            errors=errors,
            warnings=warns,
            data_shape=(len(df), len(df.columns)),
            detected_simulation_runs=simulation_runs,
            sample_range=sample_range
        )

    except Exception as e:
        return CSVValidationResponse(
            is_valid=False,
            errors=[f'Failed to process file: {str(e)}'],
            warnings=[],
        )


@app.get('/health', response_model=HealthResponse)
async def health_check():
    """
    Provides comprehensive health status information for the anomaly detection service.

    This endpoint performs health checks on critical service components to determine
    overall system availability and readiness. It evaluates the status of the machine
    learning model, Large Language Model service, and other dependencies to provide
    accurate service health reporting for monitoring systems, load balancers, and
    operational dashboards.

    The health check uses a tiered status system where 'healthy' indicates all
    components are operational, 'degraded' means core functionality is available
    but some features may be limited, and 'unhealthy' indicates critical failures
    that prevent normal operation.

    Returns:
        HealthResponse: Comprehensive health status including:
            - Overall service status (healthy/degraded/unhealthy)
            - Timestamp of the health check
            - ML model availability status
            - LLM service availability status

    Note:
        - Returns 'healthy' only when both ML model and LLM are available
        - Returns 'degraded' when ML model is loaded but LLM is unavailable
        - Returns 'unhealthy' when critical components fail or exceptions occur
        - All exceptions are logged for operational monitoring
        - Uses ISO timestamp format for consistent time representation
    """
    try:
        model_loaded = pipeline is not None

        llm_available = model_loaded and hasattr(pipeline, 'llm') and pipeline.llm is not None

        if model_loaded and llm_available:
            overall_status = 'healthy'
        else:
            overall_status = 'degraded'

        return HealthResponse(
            status=overall_status,
            timestamp=datetime.now().isoformat(),
            model_loaded=model_loaded,
            llm_available=llm_available,
        )

    except Exception as e:
        api_logger.error(f'Health check failed: {str(e)}')
        return HealthResponse(
            status='unhealthy',
            timestamp=datetime.now().isoformat(),
            model_loaded=False,
            llm_available=False,
        )


@app.get('/info', response_model=ModelInfo)
async def model_info():
    """
    Provides comprehensive information about the ML model configuration and requirements.

    This endpoint returns detailed metadata about the anomaly detection system including
    the machine learning model specifications, feature requirements, LLM configuration,
    and expected data formats. This information is essential for API consumers to
    understand system capabilities, prepare compatible data, and integrate effectively
    with the service.

    The endpoint serves as a self-documenting reference for the API, helping users
    understand what data formats are expected, how many features the model uses,
    and what underlying technologies power the analysis and explanation capabilities.

    Returns:
        ModelInfo: Comprehensive model and system information including:
            - ML model type and architecture details
            - Total number of input features expected
            - LLM model used for generating explanations
            - Training dataset description and characteristics
            - Detailed CSV format requirements and specifications

    Raises:
        HTTPException:
            - 503 if pipeline is not initialized
            - 500 if unable to retrieve model information

    Note:
        - Requires pipeline to be properly initialized and available
        - Feature count includes both original and engineered features
        - Expected format specification helps users prepare compatible data
        - Training data description provides context for model applicability
        - All errors are logged for operational monitoring
    """
    try:
        if pipeline is None:
            raise HTTPException(status_code=503, detail='Pipeline not initialized')

        expected_format = {
            'file_format': 'CSV',
            'encoding': 'UTF-8',
            'minimum_rows': 3,
            'required_columns': ['sample', 'simulationRun'],
            'feature_variables': {
                'measured': 'xmeas_1 to xmeas_41 (41 variables)',
                'manipulated': 'xmv_1 to xmv_11 (11 variables)'
            },
            'data_types': 'All process variables must be numeric',
            'temporal_requirement': 'Data must represent 3 consecutive time points'
        }

        return ModelInfo(
            model_type='XGBoost Classifier with SHAP explanations',
            features_count=len(pipeline.all_features),
            llm_model=pipeline.llm.model,
            training_data='Tennessee Eastman Process Dataset (50 simulations for each fault type, 3 fault types in total)',
            expected_csv_format=expected_format,
        )

    except Exception as e:
        api_logger.error(f'Error getting model info: {str(e)}')
        raise HTTPException(status_code=500, detail=f'Failed to get model info: {str(e)}')


@app.get('/')
async def root():
    """
    Root endpoint providing API overview and navigation information.

    This endpoint serves as the main entry point for the Tennessee Eastman Process
    Anomaly Detection API, providing users with essential information about available
    functionality, endpoints, and requirements. It acts as a discovery endpoint
    that helps new users understand the API structure and capabilities without
    needing to consult external documentation.

    The response includes version information, input format requirements, and a
    comprehensive list of available endpoints with descriptions, making it easy
    for developers to understand and integrate with the service.

    Returns:
        dict: API information including:
            - Service description and purpose
            - Current API version
            - Links to documentation and health endpoints
            - Input format requirements summary
            - Complete endpoint directory with descriptions

    Note:
        - Serves as API discovery and self-documentation endpoint
        - Provides quick reference for all available functionality
        - Includes input format requirements for user guidance
        - Links to interactive documentation at /docs
        - No authentication required for public API information
    """
    r = {
        'message': 'Tennessee Eastman Process Anomaly Detection API (CSV-based)',
        'version': '1.0.0',
        'documentation': '/docs',
        'health': '/health',
        'input_format': 'CSV files with 3+ consecutive time points',
        'endpoints': {
            'analyze': 'POST /analyze - Main anomaly analysis endpoint (CSV upload)',
            'validate-csv': 'POST /validate-csv - Validate CSV format before analysis',
            'health': 'GET /health - Service health check',
            'info': 'GET /info - Model and format information'
        }
    }
    return r


@app.post('/analyze', response_model=AnalysisResponse)
async def analyze_csv_data(
    file: UploadFile = File(..., description='CSV file with 3 consecutive time points'),
    simulation_run: Optional[Union[int, str, None]] = Form(default=None, description='Simulation run to analyze (if multiple in file)'),
    target_sample: Optional[Union[int, str, None]] = Form(default=None, description='Target sample to analyze'),
):
    """
    Performs comprehensive anomaly detection analysis on uploaded Tennessee Eastman Process data.

    This endpoint serves as the main analysis functionality of the API, accepting CSV files
    containing Tennessee Eastman Process time series data and returning detailed anomaly
    detection results. The analysis includes binary predictions, confidence scores,
    feature importance rankings, and human-readable explanations generated by the
    integrated LLM component.

    The endpoint handles the complete analysis pipeline from data validation through
    feature engineering, model inference, SHAP analysis for explainability, and
    natural language explanation generation. It supports analysis of specific simulation
    runs and target samples within multi-experiment datasets.

    Args:
        file (UploadFile): CSV file containing TEP time series data with required
            columns and at least 3 consecutive time points for temporal analysis.
        simulation_run (Optional[Union[int, str, None]]): Specific simulation run
            identifier to analyze when multiple runs are present in the data.
            If None, uses the first available run.
        target_sample (Optional[Union[int, str, None]]): Specific time sample
            to focus analysis on. If None, analyzes the final time point in
            the sequence.

    Returns:
        AnalysisResponse: Comprehensive analysis results including:
            - Binary anomaly prediction (0=normal, 1=anomaly)
            - Model confidence score
            - Top influential features with importance scores
            - Human-readable explanation from LLM
            - Processing metadata and timing information

    Raises:
        HTTPException:
            - 503 if analysis pipeline is not available
            - 400 for invalid file format, encoding, or validation errors
            - 500 for unexpected analysis failures

    Note:
        - Validates CSV format before processing
        - Logs warnings for non-critical validation issues
        - Measures and returns processing time for performance monitoring
        - Extracts analysis results from pipeline attributes after execution
        - Supports both integer and string parameter inputs with normalization
    """
    start_time = datetime.now()

    if simulation_run == '':
        simulation_run = None
    if target_sample == '':
        target_sample = None

    try:
        if pipeline is None:
            raise HTTPException(status_code=503, detail='Analysis pipeline not available. Please try again later.')

        if not file.filename.lower().endswith('.csv'):
            raise HTTPException(status_code=400, detail='File must be a CSV file')

        api_logger.info(f'Received CSV file: {file.filename} ({file.size} bytes)')

        try:
            contents = await file.read()
            csv_string = contents.decode('utf-8')
            df = pd.read_csv(StringIO(csv_string))

            api_logger.info(f'CSV loaded successfully: {df.shape} shape')

        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail='Invalid CSV encoding. Please use UTF-8 encoding.')

        except pd.errors.EmptyDataError:
            raise HTTPException(status_code=400, detail='CSV file is empty')

        except Exception as e:
            raise HTTPException(status_code=400, detail=f'Failed to parse CSV: {str(e)}')

        is_valid, errors, warns = validate_csv_data(df)

        if not is_valid:
            raise HTTPException(status_code=400, detail=f"CSV validation failed: {'; '.join(errors)}")

        for warning in warns:
            api_logger.warning(warning)

        api_logger.info(f'Analysis parameters: simulation_run={simulation_run}, target_sample={target_sample}')

        explanation_text = pipeline.analyze_sample(
            data=df,
            simulation_run=simulation_run,
            target_sample=target_sample,
        )

        simulation_run = getattr(pipeline, 'simulation_run', 0)
        target_sample = getattr(pipeline, 'target_sample', 0)
        impactful_features = getattr(pipeline, 'impactful_features', [])
        prediction = getattr(pipeline, 'prediction', 0)
        confidence = getattr(pipeline, 'confidence', 0.0)

        formatted_features = format_shap_features(impactful_features)

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        response = AnalysisResponse(
            prediction=int(prediction),
            confidence=round(float(confidence), 4),
            important_features=formatted_features,
            explanation=explanation_text,
            timestamp=datetime.now().isoformat(),
            processing_time_ms=round(processing_time, 2),
            input_rows_count=len(df),
            simulation_run=simulation_run,
            target_sample=target_sample,
        )

        api_logger.info(f'Analysis completed successfully in {processing_time:.2f}ms')
        return response

    except HTTPException:
        raise

    except Exception as e:
        api_logger.error(f'Unexpected error in analysis: {str(e)}')
        raise HTTPException(status_code=500, detail=f'Analysis failed: {str(e)}')


def run_api_server(app, host: str = '127.0.0.1', port: int = 8000, reload: bool = False):
    """
    Starts the FastAPI server for the Tennessee Eastman Process anomaly detection service.

    This function configures and launches the uvicorn ASGI server with the specified
    parameters, providing a convenient wrapper around uvicorn.run() with appropriate
    defaults for the anomaly detection API. It displays helpful startup information
    including server URLs and key endpoints to assist with development and deployment.

    The function serves as the main entry point for running the API server in both
    development and production environments, with reload functionality for development
    and configurable host/port settings for different deployment scenarios.

    Args:
        app: The FastAPI application instance to serve. Should be properly configured
            with all routes, middleware, and lifespan management.
        host (str, optional): The host address to bind the server to. Use '0.0.0.0'
            for external access or '127.0.0.1' for localhost only. Defaults to '127.0.0.1'.
        port (int, optional): The port number to listen on. Should be available
            and not conflicting with other services. Defaults to 8000.
        reload (bool, optional): Whether to enable auto-reload on code changes.
            Useful for development but should be False in production. Defaults to False.

    Returns:
        None: This function runs the server and blocks until shutdown.

    Note:
        - Displays startup information with key URLs for user convenience
        - Uses 'info' log level for appropriate production logging
        - Server runs synchronously and blocks the calling thread
        - Reload functionality watches for file changes in development
        - Consider using process managers like gunicorn for production deployment
    """
    print(f'Starting CSV-based Anomaly Detection API server...')
    print(f'Server will be available at: http://{host}:{port}')
    print(f'API documentation: http://{host}:{port}/docs')
    print(f'Upload CSV files at: http://{host}:{port}/analyze')

    uvicorn.run(
        app=app,
        host=host,
        port=port,
        reload=reload,
        log_level='info',
    )


if __name__ == "__main__":
    nest_asyncio.apply()
    run_api_server(app)