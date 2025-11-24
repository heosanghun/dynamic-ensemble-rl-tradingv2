# Code Structure and Organization

## Module Overview

### Data Processing (`src/data/`)

**data_processor.py**
- `MarketDataHandler`: Main data loading and preprocessing
- Handles OHLCV data validation and Walk-Forward splitting

**feature_extractor.py**
- `TechnicalFeatureExtractor`: Calculates 15 technical indicators
- Implements rolling window normalization to prevent look-ahead bias

**candlestick_generator.py**
- `CandlestickGenerator`: Generates 224x224 candlestick images
- Extracts visual features using ResNet-18 CNN

**news_sentiment.py**
- `NewsSentimentExtractor`: Processes news sentiment data
- Aggregates sentiment scores over 24-hour windows

**feature_fusion.py**
- `FeatureFusion`: Combines all features into unified state vector
- Implements S_t = concatenate(F_visual, F_tech, F_senti)

### Regime Classification (`src/regime/`)

**ground_truth.py**
- `RegimeGroundTruth`: Generates labels from SMA-50 slope
- Implements Eq. 2 and Eq. 3 from paper

**regime_classifier.py**
- `RegimeClassifier`: XGBoost-based classifier
- Implements confidence-based selection (Eq. 4)

### Trading Environment (`src/env/`)

**trading_env.py**
- `MultiRegimeTradingEnv`: Gymnasium-compatible environment
- Supports three regime types with different reward functions

**rewards.py**
- `RegimeRewardCalculator`: Calculates regime-specific rewards
- Implements Bull, Bear, and Sideways reward functions

### Agent Management (`src/agents/`)

**ppo_agent.py**
- `PPOAgent`: Wrapper for Stable Baselines3 PPO
- Provides policy distribution extraction

**pool.py**
- `PPOAgentPool`: Manages 5 agents per regime
- Ensures diversity through different random seeds

**agent_manager.py**
- `HierarchicalAgentManager`: Manages 3 regime-specific pools
- Coordinates training and inference across pools

### Ensemble (`src/ensemble/`)

**weighting.py**
- `DynamicWeightCalculator`: Calculates dynamic weights
- Implements Eq. 6 (Softmax with temperature)

**ensemble_trader.py**
- `EnsembleTrader`: Aggregates policies and selects actions
- Implements Eq. 7 and Eq. 8

### Backtesting (`src/backtest/`)

**backtester.py**
- `Backtester`: Walk-Forward backtesting engine
- Simulates realistic trading with transaction costs

**metrics.py**
- `PerformanceMetrics`: Calculates all performance metrics
- Supports comparison with benchmarks

### Visualization (`src/visualization/`)

**plotter.py**
- `TradingPlotter`: Creates visualization plots
- Generates portfolio, regime, and weight visualizations

### Utilities (`src/utils/`)

**logger.py**
- `setup_logger`: Centralized logging configuration

**seed.py**
- `set_seed`: Reproducibility management
- `get_agent_seeds`: Generate seeds for multiple agents

**helpers.py**
- Utility functions for data conversion and safe operations

## Execution Scripts (`scripts/`)

**train.py**
- Trains regime classifier and PPO agents
- Supports component-wise training

**main.py**
- Main execution script implementing Algorithm 1
- Complete trading loop with all components

**evaluate.py**
- Runs backtesting and performance evaluation
- Generates visualization plots

**quick_start.py**
- Quick testing script for component validation
- Uses synthetic data for testing

**example_usage.py**
- Example usage demonstrations
- Shows how to use main components

## Configuration (`config/`)

**config.yaml**
- Main system configuration
- Data paths, training parameters, regime settings

**hyperparameters.yaml**
- Model hyperparameters from paper
- XGBoost, PPO, and ensemble parameters

**paths.yaml**
- File and directory paths
- Model and result storage locations

## Testing (`tests/`)

**test_data_processor.py**
- Tests for data loading and processing

**test_regime_classifier.py**
- Tests for regime classification

**test_trading_env.py**
- Tests for trading environment and rewards

**test_ensemble.py**
- Tests for ensemble weighting and aggregation

## Key Design Patterns

### Object-Oriented Design
- All major components are classes with clear interfaces
- Encapsulation of functionality within modules

### Configuration-Driven
- All parameters configurable via YAML files
- No hardcoded values in code

### Reproducibility
- Random seed management throughout
- Deterministic operations where possible

### Error Handling
- Comprehensive error checking and logging
- Graceful degradation when optional components unavailable

### Modularity
- Clear separation of concerns
- Easy to extend or modify individual components

## Code Quality Standards

- Type hints for all function signatures
- Docstrings following NumPy style
- No AI-generated markers in comments
- Professional code style throughout
- Comprehensive error handling

