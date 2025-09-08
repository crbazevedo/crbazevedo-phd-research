"""
Experiment Configuration Module

This module provides configuration management for experiments, ensuring
alignment with thesis parameters while maintaining flexibility for different
experimental scenarios.
"""

from typing import Dict, Any, Optional, List
import logging
from .thesis_parameters import ThesisParameters, get_default_thesis_parameters, AlternativeParameters

logger = logging.getLogger(__name__)


class ExperimentConfig:
    """
    Configuration manager for experiments.
    
    This class manages experiment configurations, ensuring alignment with
    thesis parameters while providing flexibility for different scenarios.
    """
    
    def __init__(self, parameter_set: Optional[ThesisParameters] = None):
        """
        Initialize experiment configuration.
        
        Args:
            parameter_set: Custom parameter set, defaults to thesis parameters
        """
        self.parameters = parameter_set or get_default_thesis_parameters()
        self.experiment_metadata = {}
        
    def get_asmsoa_config(self) -> Dict[str, Any]:
        """Get ASMS-EMOA algorithm configuration."""
        config = self.parameters.get_algorithm_config()
        config.update({
            'sbx_eta': self.parameters.SBX_ETA,
            'mutation_strength': self.parameters.MUTATION_STRENGTH,
            'mutation_probability': self.parameters.MUTATION_PROBABILITY
        })
        return config
    
    def get_portfolio_config(self) -> Dict[str, Any]:
        """Get portfolio configuration."""
        return self.parameters.get_portfolio_config()
    
    def get_anticipatory_config(self) -> Dict[str, Any]:
        """Get anticipatory learning configuration."""
        return self.parameters.get_anticipatory_config()
    
    def get_kalman_config(self) -> Dict[str, Any]:
        """Get Kalman filter configuration."""
        return self.parameters.get_kalman_config()
    
    def get_dirichlet_config(self) -> Dict[str, Any]:
        """Get Dirichlet model configuration."""
        return self.parameters.get_dirichlet_config()
    
    def get_experimental_setup(self) -> Dict[str, Any]:
        """Get experimental setup configuration."""
        return self.parameters.get_experimental_config()
    
    def get_decision_maker_config(self, dm_type: str) -> Dict[str, Any]:
        """
        Get configuration for specific decision maker type.
        
        Args:
            dm_type: Decision maker type ('Hv-DM', 'R-DM', 'M-DM')
            
        Returns:
            Configuration for the decision maker
        """
        if dm_type not in self.parameters.DECISION_MAKER_TYPES:
            raise ValueError(f"Unknown decision maker type: {dm_type}")
        
        base_config = {
            'static_anticipation_rate': self.parameters.STATIC_ANTICIPATION_RATE,
            'prediction_horizon': self.parameters.PREDICTION_HORIZON,
            'learning_rate_combination': self.parameters.LEARNING_RATE_COMBINATION
        }
        
        if dm_type == 'Hv-DM':
            base_config.update({
                'selection_criterion': 'hypervolume',
                'optimization_target': 'expected_hypervolume'
            })
        elif dm_type == 'R-DM':
            base_config.update({
                'selection_criterion': 'random',
                'optimization_target': 'pareto_frontier'
            })
        elif dm_type == 'M-DM':
            base_config.update({
                'selection_criterion': 'median',
                'optimization_target': 'weight_median'
            })
        
        return base_config
    
    def get_time_series_config(self) -> Dict[str, Any]:
        """Get time series configuration."""
        return {
            'historical_days': self.parameters.HISTORICAL_DAYS,
            'stride_days': self.parameters.STRIDE_DAYS,
            'prediction_steps': self.parameters.PREDICTION_STEPS,
            'initial_wealth': self.parameters.INITIAL_WEALTH,
            'transaction_cost_rate': self.parameters.TRANSACTION_COST_RATE
        }
    
    def get_constraint_config(self) -> Dict[str, Any]:
        """Get constraint configuration."""
        return {
            'min_cardinality': self.parameters.MIN_CARDINALITY,
            'max_cardinality': self.parameters.MAX_CARDINALITY,
            'min_weight': self.parameters.MIN_WEIGHT,
            'max_weight': self.parameters.MAX_WEIGHT,
            'feasibility_epsilon': self.parameters.FEASIBILITY_EPSILON,
            'reference_point': self.parameters.REFERENCE_POINT
        }
    
    def get_learning_config(self) -> Dict[str, Any]:
        """Get learning configuration."""
        return {
            'prediction_horizon': self.parameters.PREDICTION_HORIZON,
            'learning_rate_combination': self.parameters.LEARNING_RATE_COMBINATION,
            'min_learning_rate': self.parameters.MIN_LEARNING_RATE,
            'max_learning_rate': self.parameters.MAX_LEARNING_RATE,
            'tip_monte_carlo_samples': self.parameters.TIP_MONTE_CARLO_SAMPLES,
            'tip_similarity_threshold': self.parameters.TIP_SIMILARITY_THRESHOLD,
            'kf_window_size': self.parameters.KF_WINDOW_SIZE,
            'sliding_window_size': self.parameters.SLIDING_WINDOW_SIZE,
            'concentration_scaling': self.parameters.CONCENTRATION_SCALING
        }
    
    def get_full_config(self) -> Dict[str, Any]:
        """Get complete configuration dictionary."""
        return {
            'algorithm': self.get_asmsoa_config(),
            'portfolio': self.get_portfolio_config(),
            'anticipatory': self.get_anticipatory_config(),
            'kalman': self.get_kalman_config(),
            'dirichlet': self.get_dirichlet_config(),
            'experimental': self.get_experimental_setup(),
            'time_series': self.get_time_series_config(),
            'constraints': self.get_constraint_config(),
            'learning': self.get_learning_config(),
            'decision_makers': self.parameters.DECISION_MAKER_TYPES,
            'metadata': self.experiment_metadata
        }
    
    def set_experiment_metadata(self, metadata: Dict[str, Any]):
        """Set experiment metadata."""
        self.experiment_metadata.update(metadata)
    
    def validate_configuration(self) -> bool:
        """Validate the complete configuration."""
        return self.parameters.validate_parameters()
    
    def print_configuration(self):
        """Print the complete configuration."""
        self.parameters.print_parameters()
        
        if self.experiment_metadata:
            print("\n📋 Experiment Metadata:")
            for key, value in self.experiment_metadata.items():
                print(f"  {key}: {value}")


class ExperimentConfigManager:
    """
    Manager for different experiment configurations.
    
    This class provides easy access to different parameter sets
    for various experimental scenarios.
    """
    
    def __init__(self):
        """Initialize configuration manager."""
        self.configurations = {
            'thesis': ExperimentConfig(get_default_thesis_parameters()),
            'small_scale': ExperimentConfig(AlternativeParameters.get_small_scale_parameters()),
            'large_scale': ExperimentConfig(AlternativeParameters.get_large_scale_parameters()),
            'high_frequency': ExperimentConfig(AlternativeParameters.get_high_frequency_parameters()),
            'conservative': ExperimentConfig(AlternativeParameters.get_conservative_parameters())
        }
    
    def get_config(self, config_name: str = 'thesis') -> ExperimentConfig:
        """
        Get experiment configuration by name.
        
        Args:
            config_name: Configuration name ('thesis', 'small_scale', etc.)
            
        Returns:
            Experiment configuration
        """
        if config_name not in self.configurations:
            available = list(self.configurations.keys())
            raise ValueError(f"Unknown configuration: {config_name}. Available: {available}")
        
        return self.configurations[config_name]
    
    def list_configurations(self) -> List[str]:
        """List available configurations."""
        return list(self.configurations.keys())
    
    def create_custom_config(self, parameter_set: ThesisParameters, 
                           config_name: str = 'custom') -> ExperimentConfig:
        """
        Create custom configuration.
        
        Args:
            parameter_set: Custom parameter set
            config_name: Name for the configuration
            
        Returns:
            Custom experiment configuration
        """
        config = ExperimentConfig(parameter_set)
        self.configurations[config_name] = config
        return config
    
    def compare_configurations(self, config_names: List[str]) -> Dict[str, Any]:
        """
        Compare different configurations.
        
        Args:
            config_names: List of configuration names to compare
            
        Returns:
            Comparison dictionary
        """
        comparison = {}
        
        for name in config_names:
            if name not in self.configurations:
                logger.warning(f"Configuration {name} not found, skipping")
                continue
            
            config = self.configurations[name]
            comparison[name] = {
                'parameters': config.parameters.to_dict(),
                'metadata': config.experiment_metadata
            }
        
        return comparison


# Global configuration manager instance
config_manager = ExperimentConfigManager()


def get_experiment_config(config_name: str = 'thesis') -> ExperimentConfig:
    """
    Convenience function to get experiment configuration.
    
    Args:
        config_name: Configuration name
        
    Returns:
        Experiment configuration
    """
    return config_manager.get_config(config_name)


def create_thesis_experiment_config() -> ExperimentConfig:
    """
    Create experiment configuration with thesis parameters.
    
    Returns:
        Experiment configuration with thesis parameters
    """
    return get_experiment_config('thesis')


def create_small_scale_experiment_config() -> ExperimentConfig:
    """
    Create experiment configuration for small-scale testing.
    
    Returns:
        Experiment configuration for small-scale testing
    """
    return get_experiment_config('small_scale')


def create_large_scale_experiment_config() -> ExperimentConfig:
    """
    Create experiment configuration for large-scale experiments.
    
    Returns:
        Experiment configuration for large-scale experiments
    """
    return get_experiment_config('large_scale')


def validate_all_configurations() -> Dict[str, bool]:
    """
    Validate all available configurations.
    
    Returns:
        Dictionary mapping configuration names to validation results
    """
    results = {}
    
    for name, config in config_manager.configurations.items():
        results[name] = config.validate_configuration()
    
    return results


def print_all_configurations():
    """Print all available configurations."""
    print("=" * 80)
    print("AVAILABLE EXPERIMENT CONFIGURATIONS")
    print("=" * 80)
    
    for name, config in config_manager.configurations.items():
        print(f"\n📋 Configuration: {name.upper()}")
        print("-" * 40)
        config.print_configuration()
    
    print("=" * 80)
