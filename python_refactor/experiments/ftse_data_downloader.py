#!/usr/bin/env python3
"""
FTSE Data Downloader
Downloads recent FTSE 100 data from Yahoo Finance
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, timedelta
import logging
import time
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FTSEDataDownloader:
    """Downloads and manages FTSE 100 data"""
    
    def __init__(self, output_dir="ftse_recent_data"):
        self.output_dir = output_dir
        self.ftse_symbols = self._get_ftse_symbols()
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _get_ftse_symbols(self):
        """Get FTSE 100 symbols with Yahoo Finance equivalents"""
        # FTSE 100 symbols with their Yahoo Finance equivalents
        # Some symbols might need .L suffix for London exchange
        ftse_symbols = [
            # Major FTSE 100 companies
            "AAPL.L", "MSFT.L", "GOOGL.L", "AMZN.L", "TSLA.L",  # Tech
            "JNJ.L", "PG.L", "UNH.L", "HD.L", "MA.L",           # Consumer/Healthcare
            "JPM.L", "BAC.L", "WFC.L", "GS.L", "MS.L",          # Financial
            "XOM.L", "CVX.L", "COP.L", "EOG.L", "SLB.L",        # Energy
            "PFE.L", "ABBV.L", "TMO.L", "ABT.L", "DHR.L",       # Healthcare
            "KO.L", "PEP.L", "WMT.L", "COST.L", "TGT.L",        # Consumer
            "V.L", "NFLX.L", "DIS.L", "CMCSA.L", "VZ.L",        # Media/Telecom
            "ADBE.L", "CRM.L", "ORCL.L", "INTC.L", "AMD.L",     # Tech
            "NKE.L", "MCD.L", "SBUX.L", "YUM.L", "CMG.L",       # Consumer
            "BA.L", "CAT.L", "DE.L", "MMM.L", "GE.L",           # Industrial
            "LLY.L", "MRK.L", "BMY.L", "GILD.L", "REGN.L",      # Pharma
            "UNP.L", "CSX.L", "NSC.L", "KSU.L", "CP.L",         # Transportation
            "SPG.L", "PLD.L", "EQIX.L", "CCI.L", "AMT.L",       # Real Estate
            "BLK.L", "SCHW.L", "AXP.L", "C.L", "USB.L",         # Financial
            "DUK.L", "SO.L", "D.L", "AEP.L", "XEL.L",           # Utilities
            "T.L", "TMUS.L", "CHTR.L", "CME.L", "ICE.L",        # Telecom/Exchange
            "BDX.L", "CI.L", "HUM.L", "ANTM.L", "CNC.L",        # Healthcare
            "TJX.L", "ROST.L", "ULTA.L", "LVS.L", "MGM.L",      # Consumer/Gaming
            "BKNG.L", "MAR.L", "HLT.L", "AAL.L", "DAL.L",       # Travel
            "FDX.L", "UPS.L", "LMT.L", "RTX.L", "GD.L",         # Defense/Logistics
            "HON.L", "ITW.L", "ETN.L", "EMR.L", "ROK.L",        # Industrial
            "APD.L", "LIN.L", "ECL.L", "APTV.L", "BLL.L",       # Materials
            "NEE.L", "D.L", "SRE.L", "WEC.L", "AEE.L",          # Utilities
            "COF.L", "DFS.L", "SYF.L", "ALLY.L", "KEY.L",       # Financial
            "PNC.L", "TFC.L", "RF.L", "HBAN.L", "FITB.L",       # Regional Banks
            "K.L", "GIS.L", "HSY.L", "SJM.L", "CAG.L",          # Food
            "KMB.L", "CL.L", "EL.L", "ULTA.L", "NKE.L",         # Consumer
            "MCD.L", "SBUX.L", "YUM.L", "CMG.L", "DPZ.L",       # Restaurants
            "WMT.L", "TGT.L", "COST.L", "HD.L", "LOW.L",        # Retail
            "AMZN.L", "EBAY.L", "ETSY.L", "SHOP.L", "BABA.L",   # E-commerce
        ]
        
        # Remove duplicates and limit to 100
        unique_symbols = list(dict.fromkeys(ftse_symbols))[:100]
        logger.info(f"Prepared {len(unique_symbols)} FTSE symbols")
        return unique_symbols
    
    def download_ftse_data(self, start_date="2015-01-01", end_date=None, max_retries=3):
        """Download FTSE data with retry logic"""
        
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        logger.info(f"Downloading FTSE data from {start_date} to {end_date}")
        
        successful_downloads = []
        failed_downloads = []
        
        for i, symbol in enumerate(self.ftse_symbols):
            logger.info(f"Downloading {symbol} ({i+1}/{len(self.ftse_symbols)})")
            
            for attempt in range(max_retries):
                try:
                    # Download data
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(start=start_date, end=end_date)
                    
                    if len(data) > 0:
                        # Save to file
                        filename = f"{symbol.replace('.L', '')}_data.csv"
                        filepath = os.path.join(self.output_dir, filename)
                        data.to_csv(filepath)
                        
                        successful_downloads.append(symbol)
                        logger.info(f"Successfully downloaded {symbol}: {len(data)} rows")
                        break
                    else:
                        logger.warning(f"No data for {symbol}, attempt {attempt + 1}")
                        
                except Exception as e:
                    logger.warning(f"Failed to download {symbol}, attempt {attempt + 1}: {e}")
                    if attempt == max_retries - 1:
                        failed_downloads.append(symbol)
                    
                    # Wait before retry
                    time.sleep(1)
            
            # Small delay between downloads to avoid rate limiting
            time.sleep(0.5)
        
        logger.info(f"Download completed: {len(successful_downloads)} successful, {len(failed_downloads)} failed")
        
        return successful_downloads, failed_downloads
    
    def create_combined_dataset(self, min_data_points=1000):
        """Create a combined dataset from downloaded files"""
        
        logger.info("Creating combined dataset...")
        
        # Get all CSV files
        csv_files = glob.glob(os.path.join(self.output_dir, "*_data.csv"))
        
        if not csv_files:
            logger.error("No data files found")
            return None
        
        # Load and combine data
        all_data = []
        asset_names = []
        
        for filepath in csv_files:
            try:
                df = pd.read_csv(filepath, index_col=0, parse_dates=True)
                
                # Check if we have enough data
                if len(df) >= min_data_points:
                    # Use Adjusted Close price
                    if 'Adj Close' in df.columns:
                        asset_name = os.path.basename(filepath).replace('_data.csv', '')
                        asset_data = df[['Adj Close']].copy()
                        asset_data.columns = [asset_name]
                        
                        all_data.append(asset_data)
                        asset_names.append(asset_name)
                        logger.info(f"Added {asset_name}: {len(df)} rows")
                
            except Exception as e:
                logger.warning(f"Error processing {filepath}: {e}")
                continue
        
        if not all_data:
            logger.error("No valid data files found")
            return None
        
        # Merge all assets
        combined_data = all_data[0]
        for asset_data in all_data[1:]:
            combined_data = combined_data.merge(asset_data, left_index=True, right_index=True, how='inner')
        
        # Calculate returns
        returns = combined_data.pct_change().dropna()
        
        logger.info(f"Combined dataset created: {returns.shape[0]} rows, {returns.shape[1]} assets")
        logger.info(f"Date range: {returns.index.min()} to {returns.index.max()}")
        
        # Save combined dataset
        combined_filepath = os.path.join(self.output_dir, "ftse_combined_returns.csv")
        returns.to_csv(combined_filepath)
        logger.info(f"Combined dataset saved to {combined_filepath}")
        
        return returns
    
    def get_ftse_index_data(self, start_date="2015-01-01", end_date=None):
        """Download FTSE 100 index data for benchmarking"""
        
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        logger.info("Downloading FTSE 100 index data...")
        
        try:
            # FTSE 100 index symbol
            ticker = yf.Ticker("^FTSE")
            data = ticker.history(start=start_date, end=end_date)
            
            if len(data) > 0:
                # Save index data
                index_filepath = os.path.join(self.output_dir, "ftse_index_data.csv")
                data.to_csv(index_filepath)
                
                logger.info(f"FTSE index data downloaded: {len(data)} rows")
                return data
            else:
                logger.error("No FTSE index data available")
                return None
                
        except Exception as e:
            logger.error(f"Failed to download FTSE index data: {e}")
            return None

def main():
    """Main function to download FTSE data"""
    
    # Initialize downloader
    downloader = FTSEDataDownloader()
    
    # Download individual stock data
    successful, failed = downloader.download_ftse_data(
        start_date="2015-01-01",
        end_date="2024-12-31"
    )
    
    # Create combined dataset
    combined_data = downloader.create_combined_dataset(min_data_points=500)
    
    # Download index data
    index_data = downloader.get_ftse_index_data(
        start_date="2015-01-01",
        end_date="2024-12-31"
    )
    
    if combined_data is not None:
        logger.info("FTSE data download completed successfully!")
        logger.info(f"Combined dataset shape: {combined_data.shape}")
        logger.info(f"Date range: {combined_data.index.min()} to {combined_data.index.max()}")
    else:
        logger.error("Failed to create combined dataset")

if __name__ == "__main__":
    main() 