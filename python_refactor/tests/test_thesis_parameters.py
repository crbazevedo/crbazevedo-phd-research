"""
Unit tests for Thesis Parameters Configuration

Tests the thesis parameters configuration to ensure alignment with
the theoretical framework and experimental setup.
"""

import unittest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config.thesis_parameters import (
    ThesisParameters, DEFAULT_THESIS_PARAMETERS, AlternativeParameters
)
from config.experiment_config import (
    ExperimentConfig, ExperimentConfigManager, get_experiment_config
)


class TestThesisParameters(unittest.TestCase):
    """Test cases for ThesisParameters class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.params = ThesisParameters()
        
    def test_default_parameters(self):
        """Test default parameter values."""
        # Test ASMS-EMOA parameters
        self.assertEqual(self.params.POPULATION_SIZE, 20)
        self.assertEqual(self.params.GENERATIONS, 30)
        self.assertEqual(self.params.MUTATION_RATE, 0.3)
        self.assertEqual(self.params.CROSSOVER_RATE, 0.2)
        self.assertEqual(self.params.TOURNAMENT_SIZE, 2)
        
        # Test portfolio constraints
        self.assertEqual(self.params.MIN_CARDINALITY, 5)
        self.assertEqual(self.params.MAX_CARDINALITY, 15)
        self.assertEqual(self.params.FEASIBILITY_EPSILON, 0.99)
        self.assertEqual(self.params.REFERENCE_POINT, (0.2, 0.0))
        
        # Test Kalman filter parameters
        self.assertEqual(self.params.KF_WINDOW_SIZE, 20)
        self.assertEqual(self.params.KF_STATE_DIM, 4)
        self.assertEqual(self.params.KF_OBSERVATION_DIM, 2)
        self.assertEqual(self.params.MONTE_CARLO_SIMULATIONS, 1000)
        
        # Test anticipatory learning parameters
        self.assertEqual(self.params.PREDICTION_HORIZON, 2)
        self.assertEqual(self.params.LEARNING_RATE_COMBINATION, 0.5)
        self.assertEqual(self.params.STATIC_ANTICIPATION_RATE, 1.0)
        
        # Test experimental setup
        self.assertEqual(self.params.HISTORICAL_DAYS, 120)
        self.assertEqual(self.params.STRIDE_DAYS, 30)
        self.assertEqual(self.params.INITIAL_WEALTH, 100000.0)
        self.assertEqual(self.params.TRANSACTION_COST_RATE, 0.001)
        
    def test_parameter_validation(self):
        """Test parameter validation."""
        # Valid parameters should pass validation
        self.assertTrue(self.params.validate_parameters())
        
        # Test invalid parameters - create separate instances
        invalid_params1 = ThesisParameters()
        invalid_params1.POPULATION_SIZE = -1  # Invalid: negative
        self.assertFalse(invalid_params1.validate_parameters())
        
        invalid_params2 = ThesisParameters()
        invalid_params2.MUTATION_RATE = 1.5  # Invalid: > 1
        self.assertFalse(invalid_params2.validate_parameters())
        
        invalid_params3 = ThesisParameters()
        invalid_params3.MAX_CARDINALITY = 3  # Invalid: < MIN_CARDINALITY
        self.assertFalse(invalid_params3.validate_parameters())
        
    def test_to_dict(self):
        """Test conversion to dictionary."""
        param_dict = self.params.to_dict()
        
        # Check that all expected keys are present
        expected_keys = [
            'population_size', 'generations', 'mutation_rate', 'crossover_rate',
            'tournament_size', 'min_cardinality', 'max_cardinality',
            'feasibility_epsilon', 'reference_point', 'kf_window_size',
            'monte_carlo_simulations', 'sliding_window_size', 'concentration_scaling',
            'dirichlet_scale_factor', 'dirichlet_concentration', 'prediction_horizon',
            'learning_rate_combination', 'historical_days', 'stride_days',
            'initial_wealth', 'transaction_cost_rate', 'decision_maker_types',
            'static_anticipation_rate', 'sbx_eta', 'mutation_strength',
            'mutation_probability', 'max_history_size', 'correspondence_similarity_threshold',
            'convergence_tolerance', 'max_stagnation_generations', 'hypervolume_reference_point'
        ]
        
        for key in expected_keys:
            self.assertIn(key, param_dict)
        
        # Check some specific values
        self.assertEqual(param_dict['population_size'], 20)
        self.assertEqual(param_dict['generations'], 30)
        self.assertEqual(param_dict['mutation_rate'], 0.3)
        
    def test_configuration_methods(self):
        """Test configuration getter methods."""
        # Test algorithm config
        algo_config = self.params.get_algorithm_config()
        self.assertIn('population_size', algo_config)
        self.assertIn('generations', algo_config)
        self.assertIn('mutation_rate', algo_config)
        
        # Test portfolio config
        portfolio_config = self.params.get_portfolio_config()
        self.assertIn('min_cardinality', portfolio_config)
        self.assertIn('max_cardinality', portfolio_config)
        self.assertIn('feasibility_epsilon', portfolio_config)
        
        # Test anticipatory config
        anticipatory_config = self.params.get_anticipatory_config()
        self.assertIn('prediction_horizon', anticipatory_config)
        self.assertIn('learning_rate_combination', anticipatory_config)
        
        # Test Kalman config
        kalman_config = self.params.get_kalman_config()
        self.assertIn('window_size', kalman_config)
        self.assertIn('state_dim', kalman_config)
        self.assertIn('monte_carlo_simulations', kalman_config)
        
        # Test Dirichlet config
        dirichlet_config = self.params.get_dirichlet_config()
        self.assertIn('sliding_window_size', dirichlet_config)
        self.assertIn('concentration_scaling', dirichlet_config)
        
        # Test experimental config
        experimental_config = self.params.get_experimental_config()
        self.assertIn('historical_days', experimental_config)
        self.assertIn('stride_days', experimental_config)
        self.assertIn('decision_maker_types', experimental_config)
        
        # Test search operators config
        search_config = self.params.get_search_operators_config()
        self.assertIn('sbx_eta', search_config)
        self.assertIn('mutation_strength', search_config)
        self.assertIn('mutation_probability', search_config)
        
    def test_decision_maker_types(self):
        """Test decision maker types initialization."""
        self.assertIsNotNone(self.params.DECISION_MAKER_TYPES)
        self.assertIn('Hv-DM', self.params.DECISION_MAKER_TYPES)
        self.assertIn('R-DM', self.params.DECISION_MAKER_TYPES)
        self.assertIn('M-DM', self.params.DECISION_MAKER_TYPES)
        self.assertEqual(len(self.params.DECISION_MAKER_TYPES), 3)


class TestAlternativeParameters(unittest.TestCase):
    """Test cases for AlternativeParameters class."""
    
    def test_small_scale_parameters(self):
        """Test small-scale parameters."""
        params = AlternativeParameters.get_small_scale_parameters()
        
        # Check that small-scale parameters are different from default
        self.assertNotEqual(params.POPULATION_SIZE, DEFAULT_THESIS_PARAMETERS.POPULATION_SIZE)
        self.assertNotEqual(params.GENERATIONS, DEFAULT_THESIS_PARAMETERS.GENERATIONS)
        self.assertNotEqual(params.HISTORICAL_DAYS, DEFAULT_THESIS_PARAMETERS.HISTORICAL_DAYS)
        
        # Check specific values
        self.assertEqual(params.POPULATION_SIZE, 10)
        self.assertEqual(params.GENERATIONS, 20)
        self.assertEqual(params.HISTORICAL_DAYS, 60)
        self.assertEqual(params.STRIDE_DAYS, 15)
        self.assertEqual(params.MONTE_CARLO_SIMULATIONS, 500)
        
        # Should still be valid
        self.assertTrue(params.validate_parameters())
        
    def test_large_scale_parameters(self):
        """Test large-scale parameters."""
        params = AlternativeParameters.get_large_scale_parameters()
        
        # Check specific values
        self.assertEqual(params.POPULATION_SIZE, 50)
        self.assertEqual(params.GENERATIONS, 100)
        self.assertEqual(params.HISTORICAL_DAYS, 240)
        self.assertEqual(params.STRIDE_DAYS, 60)
        self.assertEqual(params.MONTE_CARLO_SIMULATIONS, 2000)
        
        # Should still be valid
        self.assertTrue(params.validate_parameters())
        
    def test_high_frequency_parameters(self):
        """Test high-frequency parameters."""
        params = AlternativeParameters.get_high_frequency_parameters()
        
        # Check specific values
        self.assertEqual(params.STRIDE_DAYS, 5)
        self.assertEqual(params.KF_WINDOW_SIZE, 10)
        self.assertEqual(params.SLIDING_WINDOW_SIZE, 10)
        self.assertEqual(params.PREDICTION_HORIZON, 1)
        
        # Should still be valid
        self.assertTrue(params.validate_parameters())
        
    def test_conservative_parameters(self):
        """Test conservative parameters."""
        params = AlternativeParameters.get_conservative_parameters()
        
        # Check specific values
        self.assertEqual(params.MIN_CARDINALITY, 10)
        self.assertEqual(params.MAX_CARDINALITY, 20)
        self.assertEqual(params.FEASIBILITY_EPSILON, 0.95)
        self.assertEqual(params.LEARNING_RATE_COMBINATION, 0.3)
        
        # Should still be valid
        self.assertTrue(params.validate_parameters())


class TestExperimentConfig(unittest.TestCase):
    """Test cases for ExperimentConfig class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ExperimentConfig()
        
    def test_initialization(self):
        """Test configuration initialization."""
        self.assertIsNotNone(self.config.parameters)
        self.assertIsInstance(self.config.parameters, ThesisParameters)
        self.assertEqual(self.config.experiment_metadata, {})
        
    def test_get_asmsoa_config(self):
        """Test ASMS-EMOA configuration."""
        config = self.config.get_asmsoa_config()
        
        # Check required keys
        required_keys = ['population_size', 'generations', 'mutation_rate', 
                        'crossover_rate', 'tournament_size', 'sbx_eta']
        for key in required_keys:
            self.assertIn(key, config)
        
        # Check values
        self.assertEqual(config['population_size'], 20)
        self.assertEqual(config['generations'], 30)
        self.assertEqual(config['mutation_rate'], 0.3)
        
    def test_get_portfolio_config(self):
        """Test portfolio configuration."""
        config = self.config.get_portfolio_config()
        
        # Check required keys
        required_keys = ['min_cardinality', 'max_cardinality', 'feasibility_epsilon']
        for key in required_keys:
            self.assertIn(key, config)
        
        # Check values
        self.assertEqual(config['min_cardinality'], 5)
        self.assertEqual(config['max_cardinality'], 15)
        self.assertEqual(config['feasibility_epsilon'], 0.99)
        
    def test_get_decision_maker_config(self):
        """Test decision maker configuration."""
        # Test Hv-DM
        hv_config = self.config.get_decision_maker_config('Hv-DM')
        self.assertEqual(hv_config['selection_criterion'], 'hypervolume')
        self.assertEqual(hv_config['optimization_target'], 'expected_hypervolume')
        
        # Test R-DM
        r_config = self.config.get_decision_maker_config('R-DM')
        self.assertEqual(r_config['selection_criterion'], 'random')
        self.assertEqual(r_config['optimization_target'], 'pareto_frontier')
        
        # Test M-DM
        m_config = self.config.get_decision_maker_config('M-DM')
        self.assertEqual(m_config['selection_criterion'], 'median')
        self.assertEqual(m_config['optimization_target'], 'weight_median')
        
        # Test invalid DM type
        with self.assertRaises(ValueError):
            self.config.get_decision_maker_config('Invalid-DM')
        
    def test_get_full_config(self):
        """Test full configuration."""
        config = self.config.get_full_config()
        
        # Check all required sections
        required_sections = ['algorithm', 'portfolio', 'anticipatory', 'kalman',
                           'dirichlet', 'experimental', 'time_series', 'constraints',
                           'learning', 'decision_makers', 'metadata']
        
        for section in required_sections:
            self.assertIn(section, config)
        
        # Check that sections are not empty
        for section in required_sections:
            if section != 'metadata':  # metadata can be empty
                self.assertTrue(len(config[section]) > 0)
        
    def test_set_experiment_metadata(self):
        """Test experiment metadata setting."""
        metadata = {'experiment_name': 'test', 'version': '1.0'}
        self.config.set_experiment_metadata(metadata)
        
        self.assertEqual(self.config.experiment_metadata['experiment_name'], 'test')
        self.assertEqual(self.config.experiment_metadata['version'], '1.0')
        
        # Test updating metadata
        additional_metadata = {'author': 'test_author'}
        self.config.set_experiment_metadata(additional_metadata)
        
        self.assertEqual(self.config.experiment_metadata['experiment_name'], 'test')
        self.assertEqual(self.config.experiment_metadata['author'], 'test_author')
        
    def test_validate_configuration(self):
        """Test configuration validation."""
        # Valid configuration should pass
        self.assertTrue(self.config.validate_configuration())
        
        # Test with invalid parameters
        invalid_config = ExperimentConfig()
        invalid_config.parameters.POPULATION_SIZE = -1
        self.assertFalse(invalid_config.validate_configuration())


class TestExperimentConfigManager(unittest.TestCase):
    """Test cases for ExperimentConfigManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.manager = ExperimentConfigManager()
        
    def test_get_config(self):
        """Test getting configurations."""
        # Test thesis config
        thesis_config = self.manager.get_config('thesis')
        self.assertIsInstance(thesis_config, ExperimentConfig)
        self.assertEqual(thesis_config.parameters.POPULATION_SIZE, 20)
        
        # Test small scale config
        small_config = self.manager.get_config('small_scale')
        self.assertIsInstance(small_config, ExperimentConfig)
        self.assertEqual(small_config.parameters.POPULATION_SIZE, 10)
        
        # Test invalid config
        with self.assertRaises(ValueError):
            self.manager.get_config('invalid_config')
        
    def test_list_configurations(self):
        """Test listing configurations."""
        configs = self.manager.list_configurations()
        
        expected_configs = ['thesis', 'small_scale', 'large_scale', 'high_frequency', 'conservative']
        for expected in expected_configs:
            self.assertIn(expected, configs)
        
    def test_create_custom_config(self):
        """Test creating custom configuration."""
        custom_params = AlternativeParameters.get_small_scale_parameters()
        custom_params.POPULATION_SIZE = 15  # Custom value
        
        custom_config = self.manager.create_custom_config(custom_params, 'custom_test')
        
        self.assertIsInstance(custom_config, ExperimentConfig)
        self.assertEqual(custom_config.parameters.POPULATION_SIZE, 15)
        
        # Check that it's added to configurations
        self.assertIn('custom_test', self.manager.configurations)
        
    def test_compare_configurations(self):
        """Test comparing configurations."""
        comparison = self.manager.compare_configurations(['thesis', 'small_scale'])
        
        self.assertIn('thesis', comparison)
        self.assertIn('small_scale', comparison)
        
        # Check that parameters are included
        self.assertIn('parameters', comparison['thesis'])
        self.assertIn('parameters', comparison['small_scale'])
        
        # Check that population sizes are different
        thesis_pop = comparison['thesis']['parameters']['population_size']
        small_pop = comparison['small_scale']['parameters']['population_size']
        self.assertNotEqual(thesis_pop, small_pop)


class TestConvenienceFunctions(unittest.TestCase):
    """Test cases for convenience functions."""
    
    def test_get_experiment_config(self):
        """Test get_experiment_config function."""
        config = get_experiment_config('thesis')
        self.assertIsInstance(config, ExperimentConfig)
        self.assertEqual(config.parameters.POPULATION_SIZE, 20)
        
    def test_create_thesis_experiment_config(self):
        """Test create_thesis_experiment_config function."""
        from config.experiment_config import create_thesis_experiment_config
        config = create_thesis_experiment_config()
        self.assertIsInstance(config, ExperimentConfig)
        self.assertEqual(config.parameters.POPULATION_SIZE, 20)
        
    def test_create_small_scale_experiment_config(self):
        """Test create_small_scale_experiment_config function."""
        from config.experiment_config import create_small_scale_experiment_config
        config = create_small_scale_experiment_config()
        self.assertIsInstance(config, ExperimentConfig)
        self.assertEqual(config.parameters.POPULATION_SIZE, 10)
        
    def test_validate_all_configurations(self):
        """Test validate_all_configurations function."""
        from config.experiment_config import validate_all_configurations
        results = validate_all_configurations()
        
        # All configurations should be valid
        for config_name, is_valid in results.items():
            self.assertTrue(is_valid, f"Configuration {config_name} failed validation")


if __name__ == '__main__':
    unittest.main()
