# Industrial Process Anomaly Detection with GenAI

An advanced machine learning system for detecting anomalies in industrial processes with human-readable explanations powered by Large Language Models (LLMs). This project demonstrates the complete ML engineering pipeline from exploratory data analysis to production-ready API deployment.

## Project Overview

This system analyzes Tennessee Eastman Process data to detect industrial anomalies and provides actionable insights through natural language explanations. The project integrates classical machine learning techniques with modern generative AI to create a comprehensive solution for industrial process monitoring.

### Key Features

- **High-Performance Anomaly Detection**: XGBoost classifier achieving 97.4% F1-score
- **Explainable AI**: SHAP-based feature importance with LLM-generated explanations
- **Multi-Framework LLM Evaluation**: comparative analysis of local (Hugging Face) vs API-based (Anthropic) models
- **Interactive Memory**: contextual analysis with conversation history
- **Robust Error Handling**: graceful degradation and comprehensive logging
- **Production-Ready API**: FastAPI-based REST service with comprehensive validation
- **Docker Containerization**: complete containerized deployment solution


## Technical Architecture

### Core Components

1. **Machine Learning Pipeline**
   - Feature engineering (lag, rolling, difference features)
   - StandardScaler for data normalization
   - XGBoost classifier for anomaly detection
   - SHAP explainer for feature importance analysis

2. **LLM Integration**
   - Anthropic Claude 3 Haiku for explanation generation
   - LangChain framework for conversation management
   - Structured prompt engineering with few-shot learning
   - Conversation memory with context preservation
   - Hugging Face Transformers evaluation (FLAN-T5 Base/Large models)
   - LangChain ConversationBufferWindowMemory for context preservation
   - Custom prompt templates with structured output parsing

3. **API Service**
   - FastAPI framework with automatic OpenAPI documentation
   - Pydantic models for request/response validation
   - File upload handling with comprehensive CSV validation
   - Health monitoring and model information endpoints

4. **Infrastructure**
   - Docker containerization for consistent deployment
   - Environment-based configuration management
   - Modular project structure for maintainability

## Dataset

The project uses the **Tennessee Eastman Process Simulation Dataset** from Kaggle:
- **Source**: https://www.kaggle.com/datasets/averkij/tennessee-eastman-process-simulation-dataset
- **Format**: 4 RData files (fault-free and faulty training/testing data)
- **Features**: 52 process variables (41 measured + 11 manipulated)
- **Fault Types**: Focus on faults 1, 6, and 13 representing different failure categories

## Project Structure

```
INDUSTRIAL_ANOMALY_DETECTION/
├── data/
│   ├── processed/           # Processed CSV files (most files are not in repository)
│   │   └── TEP_API_test.csv    # Example of a CSV file for quick API test
│   └── raw/                 # Original RData files (not in repository)
├── models/               # Trained models and preprocessing objects
│   ├── xgb_model.json       # XGBoost model
│   ├── scaler.pkl           # StandardScaler
│   ├── all_features.pkl     # Feature names after engineering
│   └── selected_features.pkl
├── notebooks/            # Jupyter notebooks for analysis and development
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_feature_engineering_and_modeling.ipynb
│   ├── 03_llm_experimentation.ipynb
│   └── 04_production_pipeline.ipynb
├── src/
│   ├── api/              # FastAPI application
│   │   ├── main.py          # API endpoints and application setup
│   │   └── models.py        # Pydantic models for request/response
│   ├── pipeline/         # ML pipeline implementation
│   │   └── anomaly_detection_pipeline.py
│   ├── scripts/          # Utility functions
│   └── utils/            # Helper utilities
├── Dockerfile               # Docker image configuration
├── docker-compose.yml       # Container orchestration
├── requirements-docker.txt  # Production dependencies
├── requirements-full.txt    # Dependencies for the entire project
├── .env.example             # Environment variables template
└── README.md
```

## Installation and Setup

### Prerequisites

- Python 3.10+
- Docker and Docker Compose (optional)
- Anthropic API key
- Conda (optional)

### Option 1: Local Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/DensQ42/industrial-anomaly-detection.git
   cd industrial-anomaly-detection
   ```

2. **Create and activate virtual environment** (for example using conda)
   ```bash
   conda create -n anomaly_detection python=3.10
   conda activate anomaly_detection
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements-full.txt
   ```

4. **Setup environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your Anthropic API key
   ```

5. **Download dataset**
   - Download Tennessee Eastman Process dataset from Kaggle
   - Place RData files in `data/raw/` directory
   - Run data processing notebooks (01-04) to generate required files

6. **Run the API**
   ```bash
   python src/api/main.py
   ```

### Option 2: Docker Deployment

1. **Clone the repository**
   ```bash
   git clone https://github.com/DensQ42/industrial-anomaly-detection.git
   cd industrial-anomaly-detection
   ```

2. **Setup environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your Anthropic API key
   ```

3. **Build and run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

The API will be available at:
- **API Server**: http://localhost:8000
- **Interactive Documentation**: http://localhost:8000/docs
- **API Documentation**: http://localhost:8000/redoc

## API Usage

### Endpoints

- `GET /` - API information and available endpoints
- `POST /analyze` - Main anomaly detection endpoint
- `POST /validate-csv` - CSV format validation
- `GET /health` - Service health check
- `GET /info` - Model and format information

### Example Usage

**Analyze CSV Data:**
```bash
curl -X 'POST' \
  'http://localhost:8000/analyze' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@TEP_API_test.csv;type=text/csv' \
  -F 'simulation_run=' \
  -F 'target_sample='
```

**Response Format:**
```json
{
  "prediction": 1,
  "confidence": 1,
  "important_features": [
    {
      "variable_name": "xmeas_18",
      "description": "Stripper temperature",
      "importance_score": 3.5541
    },
    {
      "variable_name": "xmeas_16",
      "description": "Stripper pressure",
      "importance_score": 2.6435
    },
    {
      "variable_name": "xmv_9",
      "description": "Stripper steam valve",
      "importance_score": 2.2002
    }
  ],
  "explanation": "STATUS: ALERT\nISSUE: Stripper system malfunction detected\nROOT CAUSE: Stripper steam valve failure or loss of steam supply\nIMMEDIATE ACTION: Manually control stripper steam valve to maintain temperature and pressure setpoints\nMONITORING: Stripper temperature and pressure trends",
  "timestamp": "2025-09-22T00:58:42.884972",
  "model_version": "1.0",
  "processing_time_ms": 1298.33,
  "input_rows_count": 3,
  "simulation_run": 1,
  "target_sample": 363
}
```

### CSV Data Requirements

Your CSV file must contain:
- **Minimum 3 consecutive time points** for temporal analysis
- **Required columns**: `sample`, `simulationRun`
- **Process variables**: `xmeas_1` to `xmeas_41`, `xmv_1` to `xmv_11`
- **Data types**: All process variables must be numeric
- **Encoding**: UTF-8

## Key Results

### Model Performance
- **XGBoost F1-Score**: 97.4%
- **Precision**: 99.3% (minimal false alarms)
- **Recall**: 95.6% (high fault detection rate)
- **False Positive Rate**: 1.1% (down from 44.3% baseline)
- **False Negative Rate**: 4.4% (down from 9.4% baseline)

### Technical Achievements
- Comprehensive feature engineering with rolling statistics and lag features
- SHAP-based explainability for transparent decision making
- LLM integration** for operator-friendly explanations
- Production-ready deployment with robust error handling
- Memory-enabled analysis for contextual insights

## Methodology

### 1. Exploratory Data Analysis
- Analyzed 55 features across different fault types
- Identified fault-specific variability patterns
- Discovered critical timing insights for fault injection
- Mutual information analysis revealed most informative features

### 2. Feature Engineering & Modeling
- Implemented temporal feature engineering (lag, rolling, diff)
- Comprehensive comparison of 5 anomaly detection approaches
- Hyperparameter optimization
- Selected XGBoost for optimal performance balance

### 3. LLM Integration
- Experimented with local FLAN-T5 models (insufficient domain knowledge) with various prompt engineering techniques
- Transitioned to Anthropic Claude 3 Haiku for superior explanation quality
- Implemented structured prompt engineering with few-shot learning
- SHAP integration for feature-level contribution analysis
- Implemented conversation LangChain chains with memory management for contextual analysis

### 4. Production Pipeline
- FastAPI-based REST API with comprehensive validation
- Docker containerization for consistent deployment
- LangChain conversation memory for contextual analysis
- Robust error handling with graceful degradation


## Limitations and Future Work

### Current Limitations
- **Memory persistence**: conversation memory limited to single session
- **Local model constraints**: hardware limitations for larger local LLMs
- **Dataset scope**: focused on 3 specific fault types

### Future Enhancements
- Integration with live process data
- MLflow integration for production monitoring




---

**Note**: This project demonstrates academic and research capabilities in industrial AI applications. Ensure proper validation and testing before deploying in actual industrial environments.