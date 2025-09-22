import joblib, shap, os, hashlib, logging
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.schema.exceptions import LangChainException
from langchain.schema.output_parser import OutputParserException
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains.conversation.base import ConversationChain
from langchain.prompts import PromptTemplate
from pydantic import ValidationError
from datetime import datetime
from typing import Tuple, List
from src.utils.paths import get_model_path
from src.scripts.feature_engineering import create_lag_features, create_diff_features, create_rolling_features
logger = logging.getLogger(__name__)


class ImprovedAnomalyDetectionPipeline:
    """
    Comprehensive anomaly detection pipeline for Tennessee Eastman Process analysis.

    This class implements a complete machine learning pipeline for detecting anomalies
    in Tennessee Eastman Process (TEP) data, combining XGBoost classification with
    SHAP explainability and LLM-generated natural language explanations. The pipeline
    handles feature engineering, model inference, explainability analysis, and
    conversational AI integration for industrial process monitoring.

    The system is designed for real-time anomaly detection in chemical processes,
    providing both quantitative predictions and human-readable explanations suitable
    for plant operators and process engineers. It maintains conversation memory for
    contextual analysis and implements caching for improved performance.

    Attributes:
        anomaly_detector (XGBClassifier): Trained XGBoost model for anomaly prediction
        explainer (shap.TreeExplainer): SHAP explainer for feature importance analysis
        scaler: Fitted data scaler for feature normalization
        all_features (list): Complete list of engineered features used by the model
        selected_features (list): Subset of original features for engineering
        llm (ChatAnthropic): Language model for generating explanations
        conversation_chain (ConversationChain): LangChain conversation with memory
        response_cache (dict): Cache for LLM responses to improve performance
        FEATURE_DESCRIPTIONS (dict): Mapping of TEP variables to human descriptions
    """
    FEATURE_DESCRIPTIONS = {
            'xmeas_1': 'A feed (stream 1)',
            'xmeas_2': 'D feed (stream 2)',
            'xmeas_3': 'E feed (stream 3)',
            'xmeas_4': 'A and C feed (stream 4)',
            'xmeas_5': 'Recycle flow (stream 8)',
            'xmeas_6': 'Reactor feed rate (stream 6)',
            'xmeas_7': 'Reactor pressure',
            'xmeas_8': 'Reactor level',
            'xmeas_9': 'Reactor temperature',
            'xmeas_10': 'Purge rate (stream 9)',
            'xmeas_11': 'Product separator temperature',
            'xmeas_12': 'Product separator level',
            'xmeas_13': 'Product separator pressure',
            'xmeas_14': 'Product separator underflow (stream 10)',
            'xmeas_15': 'Stripper level',
            'xmeas_16': 'Stripper pressure',
            'xmeas_17': 'Stripper underflow (stream 11)',
            'xmeas_18': 'Stripper temperature',
            'xmeas_19': 'Stripper steam flow',
            'xmeas_20': 'Compressor work',
            'xmeas_21': 'Reactor cooling water outlet temperature',
            'xmeas_22': 'Separator cooling water outlet temperature',
            'xmeas_23': 'A composition in reactor feed (stream 6)',
            'xmeas_24': 'B composition in reactor feed (stream 6)',
            'xmeas_25': 'C composition in reactor feed (stream 6)',
            'xmeas_26': 'D composition in reactor feed (stream 6)',
            'xmeas_27': 'E composition in reactor feed (stream 6)',
            'xmeas_28': 'F composition in reactor feed (stream 6)',
            'xmeas_29': 'A composition in purge gas (stream 9)',
            'xmeas_30': 'B composition in purge gas (stream 9)',
            'xmeas_31': 'C composition in purge gas (stream 9)',
            'xmeas_32': 'D composition in purge gas (stream 9)',
            'xmeas_33': 'E composition in purge gas (stream 9)',
            'xmeas_34': 'F composition in purge gas (stream 9)',
            'xmeas_35': 'G composition in purge gas (stream 9)',
            'xmeas_36': 'H composition in purge gas (stream 9)',
            'xmeas_37': 'D composition in product (stream 11)',
            'xmeas_38': 'E composition in product (stream 11)',
            'xmeas_39': 'F composition in product (stream 11)',
            'xmeas_40': 'G composition in product (stream 11)',
            'xmeas_41': 'H composition in product (stream 11)',
            'xmv_1': 'D feed flow valve (stream 2)',
            'xmv_2': 'E feed flow valve (stream 3)',
            'xmv_3': 'A feed flow valve (stream 1)',
            'xmv_4': 'A and C feed flow valve  (stream 4)',
            'xmv_5': 'Compressor recycle valve',
            'xmv_6': 'Purge valve (stream 9)',
            'xmv_7': 'Separator pot liquid flow valve (stream 10)',
            'xmv_8': 'Stripper liquid product flow valve (stream 11)',
            'xmv_9': 'Stripper steam valve',
            'xmv_10': 'Reactor cooling water flow',
            'xmv_11': 'Condenser cooling water flow',
            'xmv_12': 'Agitator speed',
        }

    PROMPT_TEMPLATE = """You are a Tennessee Eastman Process control engineer analyzing plant anomalies. Your role is to provide actionable technical analysis for plant operators.

CONTEXT: Tennessee Eastman is a chemical process with reactor, separator, stripper, and recycle streams producing products G and H from reactants A, C, D, E.

HISTORICAL CONTEXT (last 5 relevant analyses):
{HISTORY}

ANALYSIS FORMAT:
- STATUS: [NORMAL (less than 50% confidence) / CAUTION (between 50% and 75% confidence) / ALERT (more than 75% confidence)]
- ISSUE: Brief technical description (one sentence)
- ROOT CAUSE: Most likely physical/chemical cause (one sentence)
- IMMEDIATE ACTION: Single most critical operator step
- MONITORING: One key parameter to track

EXAMPLES:

Input: NORMAL: 15.2% anomaly probability
Output: STATUS: NORMAL - Continue routine monitoring of all process variables.

Input: CAUTION: 65.8% confidence - Key factors: Reactor temperature, Cooling water outlet temperature
Output: STATUS: CAUTION
ISSUE: Reactor thermal management deviation detected
ROOT CAUSE: Cooling water system efficiency reduction or heat duty increase
IMMEDIATE ACTION: Verify cooling water flow rates and heat exchanger performance
MONITORING: Track reactor temperature trend

Input: ALERT: 94.3% confidence - Key factors: Product separator pressure, Product separator level, Product separator temperature
Output: STATUS: ALERT
ISSUE: Product separator control system failure detected
ROOT CAUSE: Multiple control loops failing simultaneously indicating instrumentation malfunction
IMMEDIATE ACTION: Switch separator to manual control and verify pressure relief systems
MONITORING: Product separator pressure

CURRENT ANALYSIS:
{CONTEXT}
Output:"""

    def __init__(self):
        """
        Initializes the complete anomaly detection pipeline with all components.

        Loads the trained XGBoost model, SHAP explainer, data scaler, and feature
        lists from saved files. Initializes the LangChain conversation system with
        Anthropic's Claude model for generating natural language explanations.
        Sets up conversation memory and response caching for improved performance.

        Raises:
            Exception: If model files cannot be loaded or LLM initialization fails.
        """
        self.anomaly_detector = XGBClassifier(n_jobs=-1, random_state=42)
        self.anomaly_detector.load_model(str(get_model_path('xgb_model.json')))

        self.explainer = shap.TreeExplainer(self.anomaly_detector)
        self.scaler = joblib.load(get_model_path('scaler.pkl'))
        self.all_features = joblib.load(get_model_path('all_features.pkl'))
        self.selected_features = joblib.load(get_model_path('selected_features.pkl'))

        self.response_cache = {}

        load_dotenv()

        try:
            self.llm = ChatAnthropic(
                model='claude-3-haiku-20240307',
                temperature=0.3,
                max_tokens=200,
                api_key=os.getenv('ANTHROPIC_API_KEY')
            )

            self.prompt_template = PromptTemplate(
                template=self.PROMPT_TEMPLATE,
                input_variables=['CONTEXT', 'HISTORY'],
            )

            self.memory = ConversationBufferWindowMemory(
                k=3,
                memory_key='HISTORY',
                input_key='CONTEXT',
                return_messages=False,
            )

            self.conversation_chain = ConversationChain(
                llm=self.llm,
                prompt=self.prompt_template,
                memory=self.memory,
                input_key='CONTEXT',
            )

            logger.info('ImprovedAnomalyDetectionPipeline initialized successfully')

        except Exception as e:
            logger.error(f'Failed to initialize LangChain components: {str(e)}')
            raise

    def get_sequence_for_analysis(self,
                                  data: pd.DataFrame,
                                  target_sample: int = None,
                                  simulation_run: int = None) -> pd.DataFrame:
        """
        Extracts a 3-consecutive-timepoint sequence for temporal analysis.

        This method selects exactly 3 consecutive time points ending at the target
        sample for feature engineering that requires temporal context. The sequence
        provides the necessary historical context for lag features, rolling statistics,
        and difference calculations.

        Args:
            data (pd.DataFrame): Input data containing multiple time points
            target_sample (int, optional): The final time point to analyze
            simulation_run (int, optional): Specific simulation run to extract from

        Returns:
            pd.DataFrame: Filtered data with exactly 3 consecutive time points

        Note:
            - Requires target_sample >= 2 to have sufficient historical context
            - Uses class attributes if parameters are not provided
        """
        if target_sample is None:
            target_sample = self.target_sample
        if simulation_run is None:
            simulation_run = self.simulation_run

        assert data.shape[0] >= 3, 'At least 3 samples must be provided'
        assert target_sample >= 2, 'Target sample has to have at least 2 previous time steps'

        sim_data = data[data['simulationRun'] == simulation_run]
        start_idx = target_sample - 2
        end_idx = target_sample
        mask = (sim_data['sample'] >= start_idx) & (sim_data['sample'] <= end_idx)
        return sim_data[mask]

    def feature_engineering(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Applies comprehensive feature engineering to create temporal features.

        Creates lag features (1 and 2 steps), rolling statistics (3-point window),
        and first differences for the selected features. These engineered features
        capture temporal dependencies and trends essential for accurate anomaly
        detection in time series data.

        Args:
            data (pd.DataFrame): Input data with temporal structure

        Returns:
            pd.DataFrame: Data with additional engineered features
        """
        data = create_lag_features(data=data, lags=[1,2], columns=self.selected_features, group_by='simulationRun', dropna=False)
        data = create_rolling_features(data=data, window_sizes=[3], columns=self.selected_features, group_by='simulationRun', dropna=False)
        data = create_diff_features(data=data, columns=self.selected_features, group_by='simulationRun', dropna=True)
        return data

    def feature_scaling(self, data: pd.DataFrame) -> np.ndarray:
        """
        Applies fitted scaler to normalize features for model input.

        Args:
            data (pd.DataFrame): Data with engineered features

        Returns:
            np.ndarray: Scaled feature matrix ready for model prediction
        """
        X = self.scaler.transform(data[self.all_features])
        return X

    def predict_anomaly(self, X: np.ndarray) -> Tuple[int, float]:
        """
        Generates binary prediction and confidence score using XGBoost model.

        Args:
            X (np.ndarray): Scaled feature matrix

        Returns:
            tuple[int, float]: Binary prediction (0/1) and confidence probability
        """
        confidence = self.anomaly_detector.predict_proba(X)[0, 1]
        prediction = self.anomaly_detector.predict(X)[0]
        return prediction, confidence

    def get_shap_importance(self, X: np.ndarray) -> np.ndarray:
        """
        Calculates SHAP values for feature importance and explainability.

        Args:
            X (np.ndarray): Scaled feature matrix

        Returns:
            np.ndarray: SHAP values indicating feature contributions to prediction
        """
        shap_values = self.explainer(X).values
        return shap_values

    def get_most_impactful_features(self, shap_values: np.ndarray, top_n: int = 3) -> list:
        """
        Identifies the most influential features by aggregating SHAP values.

        Aggregates SHAP contributions across all engineered versions of each base
        feature (original, lag, rolling, diff) to identify which process variables
        are most important for the current prediction.

        Args:
            shap_values (np.ndarray): SHAP values for all features
            top_n (int): Number of top features to return

        Returns:
            list: Top influential features with descriptions and impact scores
        """
        base_feature_impacts = {}

        for i, feature in enumerate(self.all_features):
            impact = shap_values[0,i]
            base_feature = '_'.join(feature.split('_')[:2])
            base_description = self.FEATURE_DESCRIPTIONS[base_feature]

            if base_feature not in base_feature_impacts:
                base_feature_impacts[base_feature] = {'total_impact': 0, 'description': base_description}

            base_feature_impacts[base_feature]['total_impact'] += impact

        impactful_features = sorted(base_feature_impacts.items(), key=lambda x: x[1]['total_impact'], reverse=True)
        return impactful_features[:top_n]

    def prepare_context_data(self,
                             confidence: float = None,
                             impactful_features: List[Tuple[str, dict]] = None) -> dict:
        """
        Prepares structured context for LLM prompt based on analysis results.

        Formats the confidence level and key influential features into a structured
        context string that matches the expected prompt format for generating
        appropriate technical explanations.

        Args:
            confidence (float, optional): Prediction confidence score
            impactful_features (list, optional): Top influential features

        Returns:
            dict: Formatted context data for LLM prompt
        """
        if confidence is None:
            confidence = self.confidence
        if impactful_features is None:
            impactful_features = self.impactful_features

        if confidence >= 0.75:
            context = f'ALERT: {confidence:.1%} confidence - Key factors: '
        elif confidence >= 0.5:
            context = f'CAUTION: {confidence:.1%} confidence - Key factors: '
        else:
            context = f'NORMAL: {confidence:.1%} anomaly probability'

        factors = []
        for _, data in impactful_features:
            description = data['description']
            if confidence >= 0.5:
                factors.append(description)

        if factors:
            context += ', '.join(factors)

        logger.info(f'Context prepared: {context[:100]}...')
        return {'CONTEXT': context}

    def _validate_response(self, response_text: str) -> bool:
        """
        Validates that LLM response contains required structured fields.

        Args:
            response_text (str): LLM generated response

        Returns:
            bool: True if response contains all required fields
        """
        required_fields = ['STATUS:', 'ISSUE:', 'ROOT CAUSE:', 'IMMEDIATE ACTION:', 'MONITORING:']
        is_valid = all(field in response_text for field in required_fields)
        return is_valid

    def _get_fallback_response(self, confidence: float = None) -> str:
        """
        Provides structured fallback response when LLM is unavailable.

        Args:
            confidence (float, optional): Prediction confidence for appropriate response

        Returns:
            str: Structured fallback response based on confidence level
        """
        if confidence is None:
            confidence = self.confidence

        if confidence >= 0.75:
            responce = """STATUS: ALERT
ISSUE: Anomaly detected with high confidence
ROOT CAUSE: Analysis system temporarily unavailable - manual investigation required
IMMEDIATE ACTION: Investigate manually using plant monitoring systems
MONITORING: All critical process parameters"""
            return responce
        elif confidence >= 0.5:
            responce = """STATUS: CAUTION
ISSUE: Potential anomaly detected
ROOT CAUSE: Analysis system temporarily unavailable - verify conditions manually
IMMEDIATE ACTION: Check key process parameters for deviations
MONITORING: Process trend monitoring"""
            return responce
        else:
            responce = 'STATUS: NORMAL - Continue routine monitoring of all process variables.'
            return responce

    def _create_cache_key(self, confidence: float = None, impactful_features: list = None) -> str:
        """
        Creates unique cache key for response caching based on analysis inputs.

        Args:
            confidence (float, optional): Prediction confidence
            impactful_features (list, optional): Key influential features

        Returns:
            str: MD5 hash for cache key
        """
        if confidence is None:
            confidence = self.confidence
        if impactful_features is None:
            impactful_features = self.impactful_features

        features_str = str(sorted(f[1]['description'] for f in impactful_features))
        confidence_rounded = round(confidence, 2)

        cache_data = f'{confidence_rounded}_{features_str}'
        return hashlib.md5(cache_data.encode()).hexdigest()

    def generate_explanation(self,
                             context_data: dict,
                             confidence: float = None,
                             impactful_features: list = None) -> str:
        """
        Generates natural language explanation using LLM with caching and fallback.

        Args:
            context_data (dict): Formatted context for LLM prompt
            confidence (float, optional): Prediction confidence
            impactful_features (list, optional): Key influential features

        Returns:
            str: Structured technical explanation or fallback response
        """
        if confidence is None:
            confidence = self.confidence
        if impactful_features is None:
            impactful_features = self.impactful_features

        cache_key = self._create_cache_key(confidence, impactful_features)

        if cache_key in self.response_cache:
            logger.info(f'Cache hit for key: {cache_key[:8]}...')
            return self.response_cache[cache_key]

        try:
            logger.info('Invoking conversation chain with memory')
            response = self.conversation_chain.predict(CONTEXT=context_data['CONTEXT'])
            logger.info(f'LLM response received: {len(response)} characters')
            if self._validate_response(response):
                self.response_cache[cache_key] = response.strip()
                logger.info(f'Cached response for key: {cache_key[:8]}...')
                return response.strip()
            else:
                logger.error('Response validation failed, using fallback')
                return self._get_fallback_response(confidence)

        except (LangChainException, OutputParserException, ValidationError) as e:
            logger.error(f'LangChain specific error: {str(e)}')
            return self._get_fallback_response(confidence)

        except Exception as e:
            logger.error(f'Unexpected error: {str(e)}')
            return self._get_fallback_response(confidence)

    def _extract_action_from_response(self, response: str) -> str:
        """
        Extracts immediate action from structured LLM response.

        Args:
            response (str): Full LLM response

        Returns:
            str: Immediate action text or default action
        """
        lines = response.split('\n')
        for line in lines:
            if line.startswith('IMMEDIATE ACTION:'):
                return line.replace('IMMEDIATE ACTION:', '').strip()
        return 'Monitor process parameters'

    def format_analysis_for_memory(self,
                                   response: str,
                                   confidence: float = None,
                                   impactful_features: list = None,
                                   timestamp: str = None) -> str:
        """
        Formats analysis results for conversation memory storage.

        Args:
            response (str): LLM generated response
            confidence (float, optional): Prediction confidence
            impactful_features (list, optional): Key features
            timestamp (str, optional): Analysis timestamp

        Returns:
            str: Formatted memory entry for conversation history
        """
        if confidence is None:
            confidence = self.confidence
        if impactful_features is None:
            impactful_features = self.impactful_features
        if timestamp is None:
            timestamp = datetime.now().strftime('%H:%M')

        features = [data['description'] for _, data in impactful_features[:3]]
        action = self._extract_action_from_response(response)
        memory = f"Time: {timestamp} | Confidence: {confidence:.1%} | Features: {', '.join(features)} | Action taken: {action}"
        return memory

    def analyze_sample(self,
                       data: pd.DataFrame,
                       simulation_run: int = None,
                       target_sample: int = None) -> str:
        """
        Performs complete end-to-end anomaly analysis with explanation.

        Executes the full pipeline from data preprocessing through feature engineering,
        model prediction, SHAP analysis, and LLM explanation generation. Manages
        internal state and conversation memory for contextual analysis.

        Args:
            data (pd.DataFrame): Input time series data
            simulation_run (int, optional): Specific simulation to analyze
            target_sample (int, optional): Target time point for analysis

        Returns:
            str: Complete structured analysis explanation

        Note:
            - Sets internal state variables for use across pipeline components
            - Saves significant analyses to conversation memory
            - Provides error fallback for robust operation
        """
        try:
            self.simulation_run = data.iloc[0]['simulationRun'] if simulation_run is None else simulation_run
            self.target_sample = data['sample'].max() if target_sample is None else target_sample

            logger.info(f'Starting analysis for simulation_run: {self.simulation_run}, target_sample: {self.target_sample}')

            data = self.get_sequence_for_analysis(data)
            data = self.feature_engineering(data)
            X = self.feature_scaling(data)

            self.prediction, self.confidence = self.predict_anomaly(X)
            shap_values = self.get_shap_importance(X)

            self.impactful_features = self.get_most_impactful_features(shap_values)

            context_data = self.prepare_context_data()
            response = self.generate_explanation(context_data)

            if self.confidence >= 0.5:
                memory_entry = self.format_analysis_for_memory(response)
                logger.info(f'Saving to memory: {memory_entry}')

            logger.info('Analysis completed successfully')
            return response

        except Exception as e:
            logger.error(f'Error in analyze_sample: {str(e)}')
            return 'STATUS: ERROR - Analysis system temporarily unavailable. Please use manual monitoring procedures.'

    def reset_state(self) -> None:
        """
        Resets all internal state variables to prepare for new analysis.
        """
        self.prediction = None
        self.confidence = None
        self.impactful_features = None
        self.simulation_run = None
        self.target_sample = None
