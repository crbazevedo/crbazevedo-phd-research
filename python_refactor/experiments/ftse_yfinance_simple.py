#!/usr/bin/env python3
"""
Simple FTSE Data Downloader using yfinance
Sets SSL environment variable to handle certificate issues
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
import certifi
from datetime import datetime
import logging

# Set SSL certificate environment
os.environ["SSL_CERT_FILE"] = certifi.where()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_ftse_data(start_date="2005-01-01", end_date=None):
    """Download FTSE 100 data using yfinance"""
    
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    logger.info(f"Downloading FTSE 100 data from {start_date} to {end_date}")
    
    try:
        # Download FTSE 100 index data
        ftse_data = yf.download("^FTSE", start=start_date, end=end_date, progress=False)
        
        if ftse_data.empty:
            logger.error("No FTSE data downloaded")
            return None
        
        # Keep only the Close price
        closes = ftse_data["Close"].dropna()
        
        logger.info(f"Downloaded {len(closes)} days of FTSE data")
        logger.info(f"Date range: {closes.index.min()} to {closes.index.max()}")
        logger.info(f"Latest close: {closes.iloc[-1]:.2f}")
        
        # Save to CSV
        closes.to_csv("ftse_close.csv", header=["Close"])
        logger.info("FTSE data saved to ftse_close.csv")
        
        return closes
        
    except Exception as e:
        logger.error(f"Error downloading FTSE data: {e}")
        return None

def download_ftse_components(start_date="2015-01-01", end_date=None, max_assets=30):
    """Download FTSE 100 component stocks"""
    
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    # FTSE 100 component symbols (major ones)
    ftse_symbols = [
        "HSBA.L", "SHEL.L", "AZN.L", "ULVR.L", "BP.L", "GSK.L", "RIO.L", "REL.L", "LSEG.L", "CRH.L",
        "PRU.L", "BHP.L", "VOD.L", "IMB.L", "LLOY.L", "BARC.L", "RKT.L", "AAL.L", "NXT.L", "SGE.L",
        "CNA.L", "MNG.L", "WPP.L", "STAN.L", "TSCO.L", "KGF.L", "MKS.L", "SMDS.L", "SSE.L", "LGEN.L",
        "PSON.L", "RKT.L", "AHT.L", "BDEV.L", "CRDA.L", "DCC.L", "EXPN.L", "FERG.L", "HLMA.L", "ICGT.L",
        "INF.L", "ITRK.L", "JD.L", "JMAT.L", "LAND.L", "MNDI.L", "OCDO.L", "PAGE.L", "PSN.L", "RTO.L"
    ]
    
    logger.info(f"Downloading {min(max_assets, len(ftse_symbols))} FTSE component stocks...")
    
    all_data = []
    successful_downloads = 0
    
    for i, symbol in enumerate(ftse_symbols[:max_assets]):
        try:
            logger.info(f"Downloading {symbol} ({i+1}/{min(max_assets, len(ftse_symbols))})")
            
            # Download stock data
            stock_data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            
            if not stock_data.empty and len(stock_data) > 100:  # At least 100 days of data
                # Use Adjusted Close for returns calculation
                adj_close = stock_data["Adj Close"].dropna()
                
                if len(adj_close) > 100:
                    # Rename column to stock symbol
                    adj_close.name = symbol.replace('.L', '')
                    all_data.append(adj_close)
                    successful_downloads += 1
                    logger.info(f"Successfully downloaded {symbol}: {len(adj_close)} days")
                else:
                    logger.warning(f"Insufficient data for {symbol}")
            else:
                logger.warning(f"No data for {symbol}")
                
        except Exception as e:
            logger.warning(f"Error downloading {symbol}: {e}")
            continue
    
    if not all_data:
        logger.error("No component stocks downloaded successfully")
        return None
    
    # Combine all stocks into a single DataFrame
    combined_data = pd.concat(all_data, axis=1, join='inner')
    
    logger.info(f"Successfully downloaded {successful_downloads} stocks")
    logger.info(f"Combined data shape: {combined_data.shape}")
    logger.info(f"Date range: {combined_data.index.min()} to {combined_data.index.max()}")
    
    # Calculate returns
    returns = combined_data.pct_change().dropna()
    
    # Save combined data
    combined_data.to_csv("ftse_components_prices.csv")
    returns.to_csv("ftse_components_returns.csv")
    
    logger.info("Component data saved to ftse_components_prices.csv and ftse_components_returns.csv")
    
    return returns

def test_ftse_download():
    """Test FTSE download with recent data"""
    
    logger.info("Testing FTSE download with recent data...")
    
    try:
        # Test with recent data
        df = yf.download("^FTSE", period="6mo", interval="1d", progress=False)
        
        if not df.empty:
            logger.info(f"Test successful! Downloaded {len(df)} days of data")
            logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
            logger.info("Last 5 days:")
            print(df.tail())
            return True
        else:
            logger.error("Test failed - no data downloaded")
            return False
            
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        return False

def main():
    """Main function to download FTSE data"""
    
    # Test download first
    logger.info("=== Testing FTSE Download ===")
    if not test_ftse_download():
        logger.error("Test failed, aborting download")
        return
    
    # Download FTSE index
    logger.info("\n=== Downloading FTSE 100 Index ===")
    ftse_index = download_ftse_data(start_date="2005-01-01")
    
    if ftse_index is not None:
        print("\nFTSE 100 Index - Last 5 days:")
        print(ftse_index.tail())
    
    # Download FTSE components
    logger.info("\n=== Downloading FTSE 100 Components ===")
    component_returns = download_ftse_components(start_date="2015-01-01", max_assets=30)
    
    if component_returns is not None:
        print(f"\nComponent Returns - Shape: {component_returns.shape}")
        print("Last 5 days of returns:")
        print(component_returns.tail())
        
        # Summary statistics
        print(f"\nSummary Statistics:")
        print(f"Mean daily return: {component_returns.mean().mean():.6f}")
        print(f"Daily volatility: {component_returns.std().mean():.6f}")
        print(f"Total trading days: {len(component_returns)}")
        print(f"Number of assets: {len(component_returns.columns)}")

if __name__ == "__main__":
    main() 