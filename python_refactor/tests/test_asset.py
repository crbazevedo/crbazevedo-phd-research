"""
Tests for asset module functionality.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os

from src.portfolio.asset import Asset, load_asset_data, calculate_returns, validate_asset_data


class TestAsset:
    """Test cases for Asset class."""
    
    def test_asset_creation(self):
        """Test Asset object creation."""
        asset = Asset("TEST001")
        assert asset.id == "TEST001"
        assert asset.historical_close_price == []
        assert asset.validation_close_price == []
        assert asset.complete_close_price == []
    
    def test_asset_repr(self):
        """Test Asset string representation."""
        asset = Asset("TEST001")
        asset.historical_close_price = [100.0, 101.0, 102.0]
        asset.validation_close_price = [102.0, 103.0]
        
        repr_str = repr(asset)
        assert "Asset" in repr_str
        assert "TEST001" in repr_str
        assert "3" in repr_str  # historical_prices=3
        assert "2" in repr_str  # validation_prices=2


class TestLoadAssetData:
    """Test cases for load_asset_data function."""
    
    @pytest.fixture
    def sample_csv_data(self):
        """Create sample CSV data for testing."""
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.02)
        
        data = {
            'Date': dates,
            'Open': prices * 0.99,
            'High': prices * 1.01,
            'Low': prices * 0.98,
            'Close': prices,
            'Volume': np.random.randint(1000, 10000, len(dates)),
            'Adj_Close': prices
        }
        return pd.DataFrame(data)
    
    def test_load_asset_data_success(self, sample_csv_data):
        """Test successful asset data loading."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_csv_data.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            training_start = datetime(2020, 1, 1)
            training_end = datetime(2020, 6, 30)
            validation_start = datetime(2020, 7, 1)
            validation_end = datetime(2020, 12, 31)
            
            asset = load_asset_data(
                temp_file, "TEST001",
                training_start, training_end,
                validation_start, validation_end
            )
            
            assert asset.id == "TEST001"
            assert len(asset.historical_close_price) > 0
            assert len(asset.validation_close_price) > 0
            assert len(asset.complete_close_price) > 0
            
            # Check that prices are reversed (as in C++ version)
            assert asset.historical_close_price[0] != sample_csv_data.iloc[0]['Adj_Close']
            
        finally:
            os.unlink(temp_file)
    
    def test_load_asset_data_file_not_found(self):
        """Test handling of missing file."""
        with pytest.raises(FileNotFoundError):
            load_asset_data(
                "nonexistent_file.csv", "TEST001",
                datetime(2020, 1, 1), datetime(2020, 6, 30),
                datetime(2020, 7, 1), datetime(2020, 12, 31)
            )
    
    def test_load_asset_data_invalid_csv(self):
        """Test handling of invalid CSV format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("Invalid,CSV,Format\n")
            f.write("1,2,3\n")
            temp_file = f.name
        
        try:
            with pytest.raises(ValueError):
                load_asset_data(
                    temp_file, "TEST001",
                    datetime(2020, 1, 1), datetime(2020, 6, 30),
                    datetime(2020, 7, 1), datetime(2020, 12, 31)
                )
        finally:
            os.unlink(temp_file)


class TestCalculateReturns:
    """Test cases for calculate_returns function."""
    
    def test_calculate_returns_normal(self):
        """Test normal return calculation."""
        prices = [100.0, 105.0, 110.0, 108.0]
        returns = calculate_returns(prices)
        
        expected = [0.05, 0.0476, -0.0182]  # Approximate values
        assert len(returns) == 3
        
        for i, (actual, expected_val) in enumerate(zip(returns, expected)):
            assert abs(actual - expected_val) < 0.001
    
    def test_calculate_returns_empty(self):
        """Test return calculation with empty price list."""
        returns = calculate_returns([])
        assert returns == []
    
    def test_calculate_returns_single_price(self):
        """Test return calculation with single price."""
        returns = calculate_returns([100.0])
        assert returns == []
    
    def test_calculate_returns_zero_price(self):
        """Test return calculation with zero price."""
        prices = [100.0, 0.0, 110.0]
        returns = calculate_returns(prices)
        assert len(returns) == 2
        assert returns[1] == 0.0  # Should handle division by zero


class TestValidateAssetData:
    """Test cases for validate_asset_data function."""
    
    def test_validate_asset_data_valid(self):
        """Test validation of valid asset data."""
        asset = Asset("TEST001")
        asset.historical_close_price = [100.0, 101.0, 102.0]
        asset.validation_close_price = [102.0, 103.0]
        
        assert validate_asset_data(asset) == True
    
    def test_validate_asset_data_empty(self):
        """Test validation of empty asset data."""
        asset = Asset("TEST001")
        assert validate_asset_data(asset) == False
    
    def test_validate_asset_data_negative_prices(self):
        """Test validation with negative prices."""
        asset = Asset("TEST001")
        asset.historical_close_price = [100.0, -101.0, 102.0]
        asset.validation_close_price = [102.0, 103.0]
        
        assert validate_asset_data(asset) == False
    
    def test_validate_asset_data_all_zeros(self):
        """Test validation with all zero prices."""
        asset = Asset("TEST001")
        asset.historical_close_price = [0.0, 0.0, 0.0]
        asset.validation_close_price = [0.0, 0.0]
        
        assert validate_asset_data(asset) == False
    
    def test_validate_asset_data_extreme_prices(self):
        """Test validation with extremely large prices."""
        asset = Asset("TEST001")
        asset.historical_close_price = [100.0, 101.0, 1e7]  # 10 million
        asset.validation_close_price = [102.0, 103.0]
        
        assert validate_asset_data(asset) == False 