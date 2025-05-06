import requests
import pandas as pd
import os
import time
import streamlit as st
import gzip
import shutil
import urllib.request
import traceback
from io import BytesIO
from datetime import datetime
from logger_config import get_logger

# Get logger for this module using centralized configuration
logger = get_logger(__name__)

class RebrickableAPI:
    """
    Class to handle interactions with the Rebrickable API and data downloads
    """
    BASE_URL = "https://rebrickable.com/api/v3/lego"
    
    # Direct download URLs for datasets
    DOWNLOAD_URLS = {
        "themes": "https://cdn.rebrickable.com/media/downloads/themes.csv.gz",
        "colors": "https://cdn.rebrickable.com/media/downloads/colors.csv.gz",
        "part_categories": "https://cdn.rebrickable.com/media/downloads/part_categories.csv.gz",
        "parts": "https://cdn.rebrickable.com/media/downloads/parts.csv.gz",
        "part_relationships": "https://cdn.rebrickable.com/media/downloads/part_relationships.csv.gz",
        "elements": "https://cdn.rebrickable.com/media/downloads/elements.csv.gz",
        "sets": "https://cdn.rebrickable.com/media/downloads/sets.csv.gz",
        "minifigs": "https://cdn.rebrickable.com/media/downloads/minifigs.csv.gz",
        "inventories": "https://cdn.rebrickable.com/media/downloads/inventories.csv.gz",
        "inventory_parts": "https://cdn.rebrickable.com/media/downloads/inventory_parts.csv.gz",
        "inventory_sets": "https://cdn.rebrickable.com/media/downloads/inventory_sets.csv.gz",
        "inventory_minifigs": "https://cdn.rebrickable.com/media/downloads/inventory_minifigs.csv.gz"
    }
    
    def __init__(self, api_key=None):
        """
        Initialize API connector with API key
        
        Args:
            api_key (str): API key for Rebrickable
        """
        self.api_key = api_key or os.getenv("REBRICKABLE_API_KEY", "")
        self.headers = {
            "Authorization": f"key {self.api_key}",
            "Accept": "application/json"
        }
        # Create data directory
        os.makedirs("./lego_data", exist_ok=True)
    
    def _make_request(self, endpoint, params=None):
        """
        Make a request to the Rebrickable API
        
        Args:
            endpoint (str): API endpoint
            params (dict): Query parameters
            
        Returns:
            dict: JSON response from API
        """
        if params is None:
            params = {}
            
        url = f"{self.BASE_URL}/{endpoint}"
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            st.error(f"HTTP Error: {e}")
            time.sleep(1)  # Rate limit handling
            return None
        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to Rebrickable API: {e}")
            return None
    
    def _fetch_all_pages(self, endpoint, params=None):
        """
        Fetch all pages for a paginated endpoint
        
        Args:
            endpoint (str): API endpoint
            params (dict): Query parameters
            
        Returns:
            list: All results from all pages
        """
        if params is None:
            params = {}
            
        all_results = []
        page = 1
        
        while True:
            params['page'] = page
            response = self._make_request(endpoint, params)
            
            if not response or 'results' not in response:
                break
                
            results = response['results']
            all_results.extend(results)
            
            if not response.get('next'):
                break
                
            page += 1
            time.sleep(0.5)  # Avoid rate limiting
            
        return all_results
    
    def _download_and_extract_dataset(self, dataset_name):
        """
        Download and extract a dataset using urllib
        
        Args:
            dataset_name (str): Name of the dataset
            
        Returns:
            DataFrame: Parsed dataset
        """
        if dataset_name not in self.DOWNLOAD_URLS:
            st.error(f"Unknown dataset: {dataset_name}")
            return pd.DataFrame()
            
        url = self.DOWNLOAD_URLS[dataset_name]
        
        # File paths
        gz_file = f'lego_data/{dataset_name}.csv.gz'
        csv_file = f'lego_data/{dataset_name}.csv'
        
        try:
            st.info(f"Downloading {dataset_name} from {url}...")
            
            # Check if directory exists
            os.makedirs('./lego_data', exist_ok=True)
            
            # Log to console only
            logger.info(f"Using urllib to download file to {gz_file}")
            
            # Download the gzipped file with additional error handling using requests
            try:
                # Log to console but not to UI
                logger.info(f"Using requests to download {url}")
                
                # Set up headers to look like a browser to avoid being blocked
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                    'Cache-Control': 'max-age=0'
                }
                
                # Try to download with full headers
                response = requests.get(url, stream=True, headers=headers)
                response.raise_for_status()  # Raise an exception for 4XX/5XX responses
                
                # Log details to console only
                logger.info(f"Response status: {response.status_code}, Content-Length: {response.headers.get('content-length')}")
                
                # Write the content to a file
                content_length = int(response.headers.get('content-length', 0))
                logger.info(f"Starting download of {content_length} bytes")
                
                # Create a progress bar for the download - keep this in UI
                download_progress = st.progress(0)
                downloaded = 0
                
                with open(gz_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if content_length > 0:
                                # Update progress bar
                                progress = int(100 * downloaded / content_length)
                                download_progress.progress(progress / 100)
                                # Only log to console, not UI
                                if progress % 10 == 0:  # Log every 10%
                                    logger.info(f"Downloaded {downloaded}/{content_length} bytes ({progress}%)")
                
                logger.info(f"Download completed: {os.path.exists(gz_file)} (File size: {os.path.getsize(gz_file) if os.path.exists(gz_file) else 'N/A'} bytes)")
            except Exception as download_error:
                # Keep errors visible in UI
                st.error(f"Error during download: {str(download_error)}")
                logger.error(f"Download error: {str(download_error)}")
                
                # Try alternative download method as fallback
                try:
                    logger.info("Trying alternative download method with curl...")
                    import subprocess
                    result = subprocess.run(['curl', '-L', '-o', gz_file, url], capture_output=True, text=True)
                    logger.info(f"curl exit code: {result.returncode}")
                    logger.info(f"curl output: {result.stdout}")
                    logger.info(f"curl error: {result.stderr}")
                    
                    if os.path.exists(gz_file) and os.path.getsize(gz_file) > 0:
                        logger.info(f"Alternative download successful: {os.path.getsize(gz_file)} bytes")
                    else:
                        raise Exception("Alternative download failed: empty or missing file")
                except Exception as alt_error:
                    st.error(f"Alternative download also failed")
                    logger.error(f"Alternative download failed: {str(alt_error)}")
                    raise download_error
            
            # Extract the gzipped file with error handling
            try:
                # Only log to console, not UI
                logger.info(f"Extracting {gz_file} to {csv_file}")
                
                # Update status message in UI
                status_msg = st.empty()
                status_msg.info(f"Extracting {dataset_name} data...")
                
                with gzip.open(gz_file, 'rb') as f_in:
                    with open(csv_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                
                logger.info(f"Extraction completed: {os.path.exists(csv_file)} (File size: {os.path.getsize(csv_file) if os.path.exists(csv_file) else 'N/A'} bytes)")
                status_msg.empty()
            except Exception as extract_error:
                st.error(f"Error during extraction: {str(extract_error)}")
                logger.error(f"Extraction error: {str(extract_error)}")
                raise
                    
            # Remove the gzipped file to save space
            os.remove(gz_file)
            logger.info(f"Removed gzipped file: {not os.path.exists(gz_file)}")
            
            # Load the CSV into a DataFrame with error handling
            try:
                logger.info(f"Loading CSV into DataFrame: {csv_file}")
                df = pd.read_csv(csv_file)
                
                # Only log details to console
                logger.info(f"DataFrame loaded with {len(df)} rows and {len(df.columns)} columns")
                logger.info(f"DOWNLOAD SUCCESS: {dataset_name} - {len(df)} rows, {len(df.columns)} columns")
                
                # Display first two rows in console log only
                if len(df) >= 2:
                    first_two_rows = df.head(2).to_dict('records')
                    logger.info(f"First two rows of {dataset_name}:")
                    for i, row in enumerate(first_two_rows):
                        logger.info(f"Row {i+1}: {row}")
            except Exception as csv_error:
                st.error(f"Error loading CSV: {str(csv_error)}")
                raise
            
            # Add download timestamp to track data freshness
            timestamp_file = f'lego_data/{dataset_name}_timestamp.txt'
            with open(timestamp_file, 'w') as f:
                f.write(datetime.now().isoformat())
            
            st.success(f"Downloaded and extracted {dataset_name} successfully: {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error downloading and extracting {dataset_name}: {str(e)}")
            st.error(f"Error downloading {dataset_name}")
            
            # Try to load from existing CSV if available
            try:
                if os.path.exists(csv_file):
                    logger.info(f"Trying to load existing file: {csv_file}")
                    df = pd.read_csv(csv_file)
                    st.info(f"Using existing {dataset_name} file: {len(df)} rows")
                    logger.info(f"Using existing {dataset_name} from file: {len(df)} rows")
                    return df
            except Exception as csv_error:
                logger.error(f"Could not load existing CSV: {str(csv_error)}")
                st.error(f"Could not load existing {dataset_name} file")
            
            # Return empty DataFrame if all else fails
            logger.error(f"Failed to download or load {dataset_name}, returning empty DataFrame")
            st.error(f"Failed to download or load {dataset_name}")
            return pd.DataFrame()
    
    def get_sets(self, params=None):
        """Get all LEGO sets"""
        return self._fetch_all_pages('sets', params)
    
    def get_themes(self, params=None):
        """Get all LEGO themes"""
        return self._fetch_all_pages('themes', params)
    
    def get_parts(self, params=None):
        """Get all LEGO parts"""
        return self._fetch_all_pages('parts', params)
    
    def get_colors(self, params=None):
        """Get all LEGO colors"""
        return self._fetch_all_pages('colors', params)
    
    def get_part_categories(self, params=None):
        """Get all LEGO part categories"""
        return self._fetch_all_pages('part_categories', params)
    
    def get_inventories(self, params=None):
        """Get all LEGO inventories"""
        return self._fetch_all_pages('inventories', params)
    
    def get_inventory_parts(self, inventory_id):
        """Get parts for a specific inventory"""
        return self._fetch_all_pages(f'inventories/{inventory_id}/parts')
    
    def get_set_inventories(self, set_num):
        """Get inventories for a specific set"""
        return self._fetch_all_pages(f'sets/{set_num}/inventories')
    
    def get_sets_by_theme(self, theme_id):
        """Get sets filtered by theme"""
        return self._fetch_all_pages('sets', {'theme_id': theme_id})
    
    def get_element_image_url(self, element_id):
        """Get image URL for a specific element"""
        return f"https://rebrickable.com/media/elements/{element_id}.jpg"
    
    def get_part_image_url(self, part_num):
        """Get image URL for a specific part"""
        return f"https://rebrickable.com/media/parts/ldraw/{part_num}.png"
    
    def download_datasets(self, cache=True, use_db=True):
        """
        Download commonly used datasets and optionally load into SQLite database
        
        Args:
            cache (bool): Whether to use cached data if available
            use_db (bool): Whether to use SQLite database instead of in-memory DataFrames
            
        Returns:
            dict or str: Dictionary of lightweight dataset metadata or database path
        """
        # Import database manager if needed
        if use_db:
            logger.info("Initializing SQLite database manager")
            from database_manager import DatabaseManager
            db_manager = DatabaseManager()
            
            # Check if database already exists and is fresh
            db_exists = db_manager.db_exists()
            db_is_fresh = db_manager.is_data_fresh() if db_exists else False
            logger.info(f"Database exists: {db_exists}, is fresh: {db_is_fresh}")
            
            if cache and db_exists and db_is_fresh:
                st.success("Using existing database from previous session")
                logger.info("Using existing SQLite database (less than 7 days old)")
                
                # Get some basic stats for logging
                try:
                    table_stats = {}
                    for table in ['themes', 'sets', 'parts', 'inventory_parts']:
                        if db_manager.get_connection().execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'").fetchone():
                            count = db_manager.get_connection().execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                            table_stats[table] = count
                    
                    logger.info(f"Database statistics: {table_stats}")
                except Exception as e:
                    logger.warning(f"Could not retrieve database statistics: {str(e)}")
                
                return {"db_manager": db_manager, "db_status": "existing"}
        
        # Define datasets to download directly - prioritize essential datasets first
        essential_datasets = ['themes', 'sets']  # Minimal set needed to start
        additional_datasets = ['parts', 'colors', 'part_categories', 'inventories', 'inventory_parts']
        
        # Combine all datasets, with essential ones first
        all_datasets = essential_datasets + additional_datasets
        
        # Initialize result container - either lightweight metadata or minimal DataFrames for essential datasets
        dataset_metadata = {}
        
        # Try to download each dataset with clear progress indication
        st.write("### Downloading LEGO Data")
        st.write("This may take a few minutes. Data will be stored in SQLite for efficient access.")
        
        progress_bar = st.progress(0)
        status_area = st.empty()
        
        # First check for existing files
        existing_files = []
        for dataset_name in all_datasets:
            csv_file = f'lego_data/{dataset_name}.csv'
            if os.path.exists(csv_file):
                existing_files.append(dataset_name)
        
        if existing_files:
            st.info(f"Found existing files for: {', '.join(existing_files)}")
        
        # Prepare database if using it
        if use_db:
            logger.info("Creating SQLite database tables")
            db_manager.create_tables()
        
        # Download each dataset with individual progress
        for i, dataset_name in enumerate(all_datasets):
            status_area.info(f"Downloading {dataset_name} ({i+1}/{len(all_datasets)})...")
            
            # Add extra details for essential datasets
            is_essential = dataset_name in essential_datasets
            if is_essential:
                status_area.warning(f"Downloading essential dataset: {dataset_name}")
            
            # Download the dataset
            df = self._download_and_extract_dataset(dataset_name)
            
            # Store in database instead of keeping in memory if using db
            if use_db and not df.empty:
                try:
                    logger.info(f"Inserting {dataset_name} into database")
                    db_manager.insert_dataframe(dataset_name, df)
                    
                    # Store only metadata rather than full DataFrame
                    dataset_metadata[dataset_name] = {
                        "rows": len(df),
                        "columns": len(df.columns),
                        "status": "success"
                    }
                    
                    # For essential datasets, keep a small sample in memory
                    if is_essential and len(df) > 0:
                        dataset_metadata[f"{dataset_name}_sample"] = df.head(10)
                    
                    # Clear DataFrame from memory
                    del df
                    
                except Exception as db_error:
                    logger.error(f"Database error for {dataset_name}: {str(db_error)}")
                    dataset_metadata[dataset_name] = {"status": "db_error", "error": str(db_error)}
            else:
                # If not using database or download failed, store status
                if df.empty:
                    dataset_metadata[dataset_name] = {"status": "failed", "rows": 0, "columns": 0}
                else:
                    # Only keep essential datasets in memory if not using db
                    if is_essential:
                        dataset_metadata[dataset_name] = df
                    else:
                        # Just store metadata for non-essential
                        dataset_metadata[dataset_name] = {
                            "rows": len(df),
                            "columns": len(df.columns),
                            "status": "success"
                        }
                        # Clear from memory
                        del df
            
            # Update progress
            progress = (i + 1) / len(all_datasets)
            progress_bar.progress(progress)
            
            # Report success or failure
            dataset_info = dataset_metadata.get(dataset_name, {})
            if isinstance(dataset_info, pd.DataFrame) or dataset_info.get("status") == "success":
                rows = len(dataset_info) if isinstance(dataset_info, pd.DataFrame) else dataset_info.get("rows", 0)
                cols = len(dataset_info.columns) if isinstance(dataset_info, pd.DataFrame) else dataset_info.get("columns", 0)
                status_area.success(f"✓ Downloaded {dataset_name}: {rows} rows, {cols} columns")
            else:
                status_area.error(f"✗ Failed to download {dataset_name}")
            
            # For essential datasets, check if we can proceed
            if is_essential:
                dataset_status = "success" if isinstance(dataset_info, pd.DataFrame) else dataset_info.get("status")
                if dataset_status != "success":
                    status_area.error(f"Critical dataset {dataset_name} missing! Some features may not work correctly.")
                
        progress_bar.empty()
        status_area.empty()
        
        # Check if we managed to get the essential datasets
        missing_essentials = []
        for name in essential_datasets:
            dataset_info = dataset_metadata.get(name, {})
            if isinstance(dataset_info, dict) and dataset_info.get("status") != "success":
                missing_essentials.append(name)
        
        if missing_essentials:
            st.warning(f"Could not load essential datasets: {', '.join(missing_essentials)}. Some features may not work correctly.")
            logger.warning(f"MISSING ESSENTIAL DATASETS: {', '.join(missing_essentials)}")
        else:
            st.success("Successfully downloaded all essential datasets!")
            logger.info("ESSENTIAL DATASETS DOWNLOAD COMPLETE")
        
        # Log a comprehensive summary of all datasets
        logger.info("========== DOWNLOAD SUMMARY ==========")
        total_rows = 0
        successful_datasets = 0
        
        summary_data = []
        for name, info in dataset_metadata.items():
            # Skip sample datasets in the summary
            if name.endswith("_sample"):
                continue
                
            if isinstance(info, pd.DataFrame):
                # It's a DataFrame
                row_count = len(info)
                col_count = len(info.columns)
                status = "✓ Success"
                total_rows += row_count
                successful_datasets += 1
                logger.info(f"Dataset: {name} | Rows: {row_count} | Columns: {col_count}")
            elif isinstance(info, dict) and info.get("status") == "success":
                # It's metadata with success status
                row_count = info.get("rows", 0)
                col_count = info.get("columns", 0)
                status = "✓ Success"
                total_rows += row_count
                successful_datasets += 1
                logger.info(f"Dataset: {name} | Rows: {row_count} | Columns: {col_count}")
            else:
                # It's a failed dataset
                row_count = 0
                col_count = 0
                status = "✗ Failed"
                logger.warning(f"Dataset: {name} | Status: EMPTY OR FAILED")
            
            summary_data.append({"Dataset": name, "Status": status, "Rows": row_count, "Columns": col_count})
        
        logger.info(f"Total datasets: {len(all_datasets)} | Successful: {successful_datasets} | Failed: {len(all_datasets) - successful_datasets}")
        logger.info(f"Total rows across all datasets: {total_rows}")
        logger.info("======================================")
        
        # Also display the summary in the UI
        st.write("### Download Summary")
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df)
        st.info(f"Total rows across all datasets: {total_rows}")
        
        # Return the database manager if using db
        if use_db:
            result = {
                "db_manager": db_manager,
                "metadata": dataset_metadata,
                "db_status": "new",
                "total_rows": total_rows
            }
            
            # Cache the database manager reference
            if cache:
                st.session_state.db_manager = db_manager
                st.session_state.dataset_metadata = dataset_metadata
                logger.info("Database manager cached in session state")
        else:
            # If not using db, return the dataset metadata/DataFrames
            result = dataset_metadata
            
            # Cache the dataset metadata
            if cache:
                st.session_state.dataset_metadata = dataset_metadata
                logger.info("Dataset metadata cached in session state")
        
        return result