import pandas as pd
import numpy as np
import streamlit as st
import traceback
from logger_config import get_logger
from rebrickable_api import RebrickableAPI

# Initialize logger for this module
logger = get_logger(__name__)

class DataProcessor:
    """
    Process and prepare Rebrickable data for analysis
    """
    def __init__(self, api=None):
        """
        Initialize data processor
        
        Args:
            api (RebrickableAPI): API connector instance
        """
        self.api = api or RebrickableAPI()
        
    def load_data(self, use_api=True, cache=True):
        """
        Load datasets from API or CSV files
        
        Args:
            use_api (bool): Whether to use API (True) or CSV files (False)
            cache (bool): Whether to use cached data
            
        Returns:
            dict: Dictionary of pandas DataFrames
        """
        if use_api:
            return self.api.download_datasets(cache)
        else:
            # Try to load from files if provided
            try:
                datasets = {}
                file_list = ['sets.csv', 'themes.csv', 'parts.csv', 'colors.csv',
                            'inventories.csv', 'inventory_parts.csv', 'part_categories.csv',
                            'elements.csv', 'inventory_minifigs.csv', 'inventory_sets.csv',
                            'minifigs.csv', 'part_relationships.csv']
                
                for file_name in file_list:
                    try:
                        datasets[file_name.split('.')[0]] = pd.read_csv(file_name)
                    except FileNotFoundError:
                        st.warning(f"Could not find {file_name}. Skipping.")
                        
                return datasets
            except Exception as e:
                st.error(f"Error loading datasets from files: {e}")
                return {}
    
    def prepare_data_for_theme_prediction(self, datasets=None, db_manager=None):
        """
        Prepare datasets for theme prediction model
        Can use either in-memory datasets or SQLite database
        
        Args:
            datasets (dict, optional): Dictionary of pandas DataFrames
            db_manager (DatabaseManager, optional): Database manager for SQLite operations
            
        Returns:
            DataFrame: Feature matrix X (parts per set)
            Series: Target vector y (theme ids)
        """
        logger.info("Starting preparation of data for theme prediction")
        
        # Check if we have database access
        using_db = db_manager is not None and hasattr(db_manager, 'query_to_dataframe')
        logger.info(f"Using database: {using_db}")
        
        # If no database and no datasets, check session state
        if not using_db and not datasets:
            logger.info("No database or datasets provided, checking session state")
            if 'db_manager' in st.session_state:
                db_manager = st.session_state.db_manager
                using_db = True
                logger.info("Found db_manager in session state")
            elif 'datasets' in st.session_state:
                datasets = st.session_state.datasets
                logger.info("Found datasets in session state")
        
        # If still no valid data source, show error
        if not using_db and (not datasets or not all(key in datasets for key in ['sets', 'themes'])):
            logger.error("Missing required datasets for theme prediction")
            st.error("Missing required datasets for theme prediction")
            return None, None
        
        try:
            # Different approach based on data source
            if using_db:
                # Use database for memory efficiency - only load what we need with SQL
                st.info("Using SQLite database for memory-efficient data processing")
                
                # Use SQL to efficiently join the data and limit what we load in memory
                # Only select subset of 50 themes and 5 sets per theme for training
                # This is faster and more memory efficient than loading everything first
                sql_query = """
                SELECT 
                    ip.part_num, ip.quantity, i.set_num, s.theme_id
                FROM 
                    inventory_parts ip
                JOIN 
                    inventories i ON ip.inventory_id = i.id
                JOIN 
                    sets s ON i.set_num = s.set_num
                WHERE 
                    s.theme_id IN (
                        SELECT theme_id 
                        FROM sets 
                        GROUP BY theme_id 
                        HAVING COUNT(*) >= 5
                        LIMIT 50
                    )
                AND
                    i.set_num IN (
                        SELECT s2.set_num 
                        FROM sets s2
                        WHERE s2.theme_id = s.theme_id
                        LIMIT 5
                    )
                """
                
                # Execute the query
                try:
                    with st.spinner("Loading data for theme prediction (efficient SQL query)..."):
                        logger.info("Executing efficient SQL query for theme prediction data")
                        logger.debug(f"SQL Query: {sql_query}")
                        merged_df = db_manager.query_to_dataframe(sql_query)
                        logger.info(f"SQL query successful, loaded {len(merged_df)} rows")
                        st.success(f"Loaded {len(merged_df)} rows for theme prediction")
                except Exception as e:
                    error_details = traceback.format_exc()
                    logger.error(f"Database query error: {str(e)}\n{error_details}")
                    st.error(f"Database query error: {e}")
                    
                    # Fallback to simple query if the complex one fails
                    logger.info("Attempting fallback to simpler queries")
                    try:
                        # Load only a small subset of the data for model training
                        logger.info("Loading sets data (limited to 1000 rows)")
                        sets_df = db_manager.query_to_dataframe("SELECT * FROM sets LIMIT 1000")
                        
                        logger.info("Loading inventories data (limited to 1000 rows)")
                        inventories_df = db_manager.query_to_dataframe("SELECT * FROM inventories LIMIT 1000")
                        
                        logger.info("Loading inventory_parts data (limited to 10000 rows)")
                        parts_df = db_manager.query_to_dataframe("SELECT * FROM inventory_parts LIMIT 10000")
                        
                        # Join in Python (less efficient but more reliable)
                        logger.info("Performing joins in Python")
                        merged_df = parts_df.merge(
                            inventories_df,
                            left_on='inventory_id',
                            right_on='id'
                        ).merge(
                            sets_df,
                            on='set_num'
                        )
                        logger.info(f"Fallback join successful, merged dataframe has {len(merged_df)} rows")
                    except Exception as e2:
                        error_details = traceback.format_exc()
                        logger.error(f"Fallback query also failed: {str(e2)}\n{error_details}")
                        st.error(f"Fallback query also failed: {e2}")
                        return None, None
            else:
                # Use in-memory datasets
                # First check if we have enough information
                if not all(key in datasets for key in ['sets', 'inventories', 'inventory_parts']):
                    st.error("Missing required datasets for theme prediction")
                    return None, None
                
                # Join inventories with inventory_parts
                merged_df = datasets['inventory_parts'].merge(
                    datasets['inventories'], 
                    left_on='inventory_id', 
                    right_on='id'
                )
                
                # Join with sets to get theme information
                merged_df = merged_df.merge(
                    datasets['sets'], 
                    on='set_num'
                )
            
            # Now we have merged_df either way (database or in-memory)
            # Create a feature matrix with one-hot encoding for parts
            # Each row is a set, each column is a part
            
            # Limit to most common parts to reduce dimensionality (memory efficiency)
            part_counts = merged_df['part_num'].value_counts()
            common_parts = part_counts[part_counts >= 10].index.tolist()
            
            # Use only the most common parts
            merged_df_filtered = merged_df[merged_df['part_num'].isin(common_parts)]
            
            # Create pivot table (efficient one-hot encoding)
            with st.spinner("Creating feature matrix (may take a moment)..."):
                pivot_df = merged_df_filtered.pivot_table(
                    index='set_num',
                    columns='part_num',
                    values='quantity',
                    aggfunc='sum',
                    fill_value=0
                )
                
                # Get theme_id for each set
                themes = merged_df_filtered[['set_num', 'theme_id']].drop_duplicates().set_index('set_num')
                
                # Ensure indices align
                X = pivot_df
                y = themes.loc[pivot_df.index, 'theme_id']
                
                st.success(f"Created feature matrix with {X.shape[0]} sets and {X.shape[1]} parts")
                
                return X, y
            
        except Exception as e:
            st.error(f"Error preparing data for theme prediction: {e}")
            return None, None
    
    def ensure_datasets_loaded(self, required_datasets, datasets=None, db_manager=None):
        """
        Ensure that required datasets are loaded, either from memory or from database
        
        Args:
            required_datasets (list): List of dataset names required
            datasets (dict, optional): Existing datasets dictionary
            db_manager (DatabaseManager, optional): Database manager for SQLite operations
            
        Returns:
            dict: Updated datasets dictionary with required datasets
        """
        if datasets is None:
            if hasattr(st, 'session_state') and 'datasets' in st.session_state:
                datasets = st.session_state.datasets
            else:
                datasets = {}
        
        # Check which datasets are missing
        missing_datasets = [ds for ds in required_datasets if ds not in datasets]
        
        if not missing_datasets:
            # All datasets are already loaded
            return datasets
            
        logger.info(f"Missing datasets: {missing_datasets}. Attempting to load from database.")
        
        # Try to get a database manager if not provided
        if db_manager is None:
            if hasattr(st, 'session_state') and 'db_manager' in st.session_state:
                db_manager = st.session_state.db_manager
                
        # If we have a database manager, try to load missing datasets
        if db_manager is not None and hasattr(db_manager, 'query_to_dataframe'):
            try:
                with st.spinner(f"Loading additional datasets: {', '.join(missing_datasets)}"):
                    for dataset_name in missing_datasets:
                        try:
                            logger.info(f"Loading {dataset_name} from database")
                            datasets[dataset_name] = db_manager.query_to_dataframe(f"SELECT * FROM {dataset_name}")
                            logger.info(f"Successfully loaded {dataset_name} with {len(datasets[dataset_name])} rows")
                        except Exception as e:
                            error_details = traceback.format_exc()
                            logger.error(f"Failed to load {dataset_name} from database: {str(e)}\n{error_details}")
                            st.error(f"Could not load {dataset_name} dataset: {str(e)}")
                
                # Update session state if available
                if hasattr(st, 'session_state'):
                    st.session_state.datasets = datasets
                    
                # Check if we now have all required datasets
                still_missing = [ds for ds in required_datasets if ds not in datasets]
                if still_missing:
                    logger.warning(f"Still missing datasets after loading attempt: {still_missing}")
                    st.warning(f"Some required datasets could not be loaded: {', '.join(still_missing)}")
                else:
                    logger.info("Successfully loaded all required datasets")
                    
            except Exception as e:
                error_details = traceback.format_exc()
                logger.error(f"Error loading datasets from database: {str(e)}\n{error_details}")
                st.error(f"Error loading datasets from database: {str(e)}")
        else:
            logger.error("No database manager available to load missing datasets")
            st.error("Required datasets are missing and no database connection is available. Please reload the data from the sidebar.")
        
        return datasets

    def get_theme_name_mapping(self, datasets):
        """
        Create mapping from theme_id to theme_name
        
        Args:
            datasets (dict): Dictionary of pandas DataFrames
            
        Returns:
            dict: Mapping from theme_id to theme_name
        """
        # Ensure the themes dataset is loaded
        datasets = self.ensure_datasets_loaded(['themes'], datasets)
        
        if 'themes' not in datasets:
            logger.warning("Themes dataset not available for mapping")
            return {}
            
        theme_map = datasets['themes'].set_index('id')['name'].to_dict()
        logger.info(f"Created theme name mapping with {len(theme_map)} themes")
        return theme_map
    
    def get_set_parts(self, set_num, datasets):
        """
        Get all parts for a specific set - tries to use database first for better performance
        
        Args:
            set_num (str): Set number
            datasets (dict): Dictionary of pandas DataFrames
            
        Returns:
            DataFrame: Parts list with quantities
        """
        logger.info(f"Getting parts for set {set_num}")
        
        # First try to get from database if available (much faster)
        if hasattr(st, 'session_state') and 'db_manager' in st.session_state:
            db_manager = st.session_state.db_manager
            try:
                # Use SQL join for better performance
                parts_query = """
                SELECT
                    ip.part_num, 
                    ip.quantity,
                    p.name as part_name,
                    p.part_cat_id,
                    c.name as color_name
                FROM
                    inventories i
                JOIN
                    inventory_parts ip ON i.id = ip.inventory_id
                JOIN
                    parts p ON ip.part_num = p.part_num
                LEFT JOIN
                    colors c ON ip.color_id = c.id
                WHERE
                    i.set_num = ?
                """
                
                logger.info(f"Executing SQL query for parts of set {set_num}")
                parts_df = db_manager.query_to_dataframe(parts_query, (set_num,))
                
                if not parts_df.empty:
                    logger.info(f"Successfully loaded {len(parts_df)} parts for set {set_num} from database")
                    return parts_df
                else:
                    logger.warning(f"No parts found for set {set_num} in database")
            except Exception as e:
                error_details = traceback.format_exc()
                logger.error(f"Error getting parts from database for set {set_num}: {str(e)}\n{error_details}")
                logger.info("Falling back to in-memory datasets")
        
        # If database approach failed or is not available, use in-memory datasets
        logger.info(f"Using memory datasets to get parts for set {set_num}")
        
        # Make sure we have all required datasets
        required_datasets = ['inventories', 'inventory_parts', 'parts']
        datasets = self.ensure_datasets_loaded(required_datasets, datasets)
        
        # Check if any required datasets are still missing
        missing_datasets = [ds for ds in required_datasets if ds not in datasets]
        if missing_datasets:
            logger.error(f"Missing datasets after loading attempt: {missing_datasets}")
            st.error(f"Error: Could not load required datasets for set parts analysis: {', '.join(missing_datasets)}")
            return pd.DataFrame()
            
        try:
            # Get inventory IDs for this set
            inventory_filter = datasets['inventories']['set_num'] == set_num
            logger.debug(f"Filtering inventories for set_num={set_num}")
            
            # Check if there are any matching inventories
            if not inventory_filter.any():
                logger.warning(f"No inventories found for set {set_num}")
                st.warning(f"No inventory found for set {set_num}")
                return pd.DataFrame()
            
            inventory_ids = datasets['inventories'][inventory_filter]['id'].tolist()
            logger.info(f"Found {len(inventory_ids)} inventories for set {set_num}: {inventory_ids}")
            
            if not inventory_ids:
                logger.warning(f"No inventory IDs found for set {set_num}")
                return pd.DataFrame()
                
            # Get parts for these inventories
            inventory_parts_filter = datasets['inventory_parts']['inventory_id'].isin(inventory_ids)
            parts_df = datasets['inventory_parts'][inventory_parts_filter]
            
            logger.info(f"Found {len(parts_df)} inventory parts for set {set_num}")
            
            if parts_df.empty:
                logger.warning(f"No parts found for set {set_num} in inventory parts")
                return pd.DataFrame()
            
            # Add part information
            try:
                parts_df = parts_df.merge(datasets['parts'], on='part_num')
                logger.info(f"Successfully merged parts data, final dataframe has {len(parts_df)} rows")
                
                return parts_df
            except Exception as e:
                error_details = traceback.format_exc()
                logger.error(f"Error merging parts data: {str(e)}\n{error_details}")
                st.error(f"Error merging parts data for set {set_num}: {str(e)}")
                return pd.DataFrame()
                
        except Exception as e:
            error_details = traceback.format_exc()
            logger.error(f"Error getting parts for set {set_num}: {str(e)}\n{error_details}")
            st.error(f"Error getting parts for set {set_num}: {str(e)}")
            return pd.DataFrame()
    
    def get_theme_sets(self, theme_id, datasets):
        """
        Get all sets for a specific theme
        
        Args:
            theme_id (int): Theme ID
            datasets (dict): Dictionary of pandas DataFrames
            
        Returns:
            DataFrame: Sets in this theme
        """
        logger.info(f"Getting sets for theme {theme_id}")
        
        # Make sure we have the sets dataset
        datasets = self.ensure_datasets_loaded(['sets'], datasets)
        
        if 'sets' not in datasets:
            logger.error(f"Missing 'sets' dataset when getting sets for theme {theme_id}")
            st.error(f"Error: 'sets' dataset not available for theme analysis")
            return pd.DataFrame()
            
        theme_sets = datasets['sets'][datasets['sets']['theme_id'] == theme_id]
        logger.info(f"Found {len(theme_sets)} sets for theme {theme_id}")
        return theme_sets
    
    def get_unique_parts_for_theme(self, theme_id, datasets):
        """
        Find parts that are unique to a specific theme
        
        Args:
            theme_id (int): Theme ID
            datasets (dict): Dictionary of pandas DataFrames
            
        Returns:
            DataFrame: Unique parts for this theme
        """
        logger.info(f"Finding unique parts for theme {theme_id}")
        
        # Make sure we have all required datasets
        required_datasets = ['sets', 'inventories', 'inventory_parts', 'parts']
        datasets = self.ensure_datasets_loaded(required_datasets, datasets)
        
        # Check if any required datasets are still missing
        missing_datasets = [ds for ds in required_datasets if ds not in datasets]
        if missing_datasets:
            logger.error(f"Missing datasets after loading attempt for unique parts: {missing_datasets}")
            st.error(f"Error: Could not load required datasets for unique parts analysis: {', '.join(missing_datasets)}")
            return pd.DataFrame()
        
        try:
            # Get all sets in this theme
            theme_sets = self.get_theme_sets(theme_id, datasets)
            if theme_sets.empty:
                logger.warning(f"No sets found for theme {theme_id}")
                st.warning(f"No sets found for theme ID {theme_id}")
                return pd.DataFrame()
                
            # Get inventory IDs for these sets
            inventory_filter = datasets['inventories']['set_num'].isin(theme_sets['set_num'])
            if not inventory_filter.any():
                logger.warning(f"No inventories found for theme {theme_id} sets")
                st.warning(f"No inventory data found for this theme's sets")
                return pd.DataFrame()
                
            inventory_ids = datasets['inventories'][inventory_filter]['id'].tolist()
            logger.info(f"Found {len(inventory_ids)} inventories for theme {theme_id}")
            
            if not inventory_ids:
                return pd.DataFrame()
                
            # Get parts for these inventories
            parts_filter = datasets['inventory_parts']['inventory_id'].isin(inventory_ids)
            if not parts_filter.any():
                logger.warning(f"No parts found in inventories for theme {theme_id}")
                st.warning(f"No parts found in this theme's sets")
                return pd.DataFrame()
                
            theme_parts = datasets['inventory_parts'][parts_filter]
            logger.info(f"Found {len(theme_parts)} parts in theme {theme_id}")
            
            # Get parts for all other themes
            other_theme_sets = datasets['sets'][datasets['sets']['theme_id'] != theme_id]
            other_inventory_ids = datasets['inventories'][
                datasets['inventories']['set_num'].isin(other_theme_sets['set_num'])
            ]['id'].tolist()
            
            other_theme_parts = pd.DataFrame() 
            if other_inventory_ids:
                other_theme_parts = datasets['inventory_parts'][
                    datasets['inventory_parts']['inventory_id'].isin(other_inventory_ids)
                ]
            
            # Find parts that are in this theme but not in others
            if other_theme_parts.empty:
                # If no other theme parts, all parts in this theme are unique
                unique_parts = theme_parts
                logger.info(f"All {len(unique_parts)} parts in theme {theme_id} are unique (no other themes have parts)")
            else:
                # Normal case - find parts not used in other themes
                unique_parts = theme_parts[
                    ~theme_parts['part_num'].isin(other_theme_parts['part_num'].unique())
                ]
                logger.info(f"Found {len(unique_parts)} parts unique to theme {theme_id}")
            
            if unique_parts.empty:
                logger.warning(f"No unique parts found for theme {theme_id}")
                st.info(f"This theme doesn't have any unique parts that aren't used in other themes")
                return pd.DataFrame()
                
            # Add part information
            try:
                unique_parts = unique_parts.merge(datasets['parts'], on='part_num')
                logger.info(f"Successfully added part details, final dataframe has {len(unique_parts)} rows")
                return unique_parts
            except Exception as e:
                error_details = traceback.format_exc()
                logger.error(f"Error merging part data: {str(e)}\n{error_details}")
                st.error(f"Error adding part details: {str(e)}")
                return pd.DataFrame()
                
        except Exception as e:
            error_details = traceback.format_exc()
            logger.error(f"Error finding unique parts for theme {theme_id}: {str(e)}\n{error_details}")
            st.error(f"Error finding unique parts: {str(e)}")
            return pd.DataFrame()
