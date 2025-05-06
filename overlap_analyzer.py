import pandas as pd
import numpy as np
import streamlit as st
import traceback
from logger_config import get_logger

# Initialize logger for this module
logger = get_logger(__name__)

class OverlapAnalyzer:
    """
    Analyze part overlap between different LEGO sets and themes
    """
    def __init__(self, data_processor):
        """
        Initialize overlap analyzer
        
        Args:
            data_processor (DataProcessor): Data processor instance
        """
        self.data_processor = data_processor
        
    def calculate_set_overlap(self, set_num1, set_num2, datasets):
        """
        Calculate the part overlap between two sets
        
        Args:
            set_num1 (str): First set number
            set_num2 (str): Second set number
            datasets (dict): Dictionary of pandas DataFrames
            
        Returns:
            dict: Overlap statistics
        """
        logger.info(f"Calculating overlap between sets {set_num1} and {set_num2}")
        
        # Check if required datasets are available
        required_datasets = ['inventories', 'inventory_parts', 'parts']
        missing_datasets = [ds for ds in required_datasets if ds not in datasets]
        if missing_datasets:
            logger.error(f"Missing required datasets: {missing_datasets}")
            return {
                'overlap_count': 0,
                'overlap_percentage1': 0,
                'overlap_percentage2': 0,
                'overlap_parts': []
            }
            
        # Get parts for each set
        try:
            parts1 = self.data_processor.get_set_parts(set_num1, datasets)
            parts2 = self.data_processor.get_set_parts(set_num2, datasets)
            
            logger.info(f"Retrieved {len(parts1)} parts for set {set_num1} and {len(parts2)} parts for set {set_num2}")
            
            if parts1.empty or parts2.empty:
                logger.warning(f"Empty parts list for one of the sets: {set_num1}={len(parts1)}, {set_num2}={len(parts2)}")
                return {
                    'overlap_count': 0,
                    'overlap_percentage1': 0,
                    'overlap_percentage2': 0,
                    'overlap_parts': []
                }
        except Exception as e:
            error_details = traceback.format_exc()
            logger.error(f"Error calculating set overlap: {str(e)}\n{error_details}")
            return {
                'overlap_count': 0,
                'overlap_percentage1': 0,
                'overlap_percentage2': 0,
                'overlap_parts': []
            }
        
        # Get unique part numbers for each set
        parts1_nums = set(parts1['part_num'].unique())
        parts2_nums = set(parts2['part_num'].unique())
        
        # Calculate overlap
        overlap_parts = parts1_nums.intersection(parts2_nums)
        
        # Get detailed information about overlapping parts
        if overlap_parts:
            overlap_details = parts1[parts1['part_num'].isin(overlap_parts)].merge(
                parts2[['part_num', 'quantity']],
                on='part_num',
                suffixes=('_set1', '_set2')
            )
        else:
            overlap_details = pd.DataFrame()
        
        return {
            'overlap_count': len(overlap_parts),
            'overlap_percentage1': len(overlap_parts) / len(parts1_nums) * 100 if parts1_nums else 0,
            'overlap_percentage2': len(overlap_parts) / len(parts2_nums) * 100 if parts2_nums else 0,
            'overlap_parts': overlap_details
        }
    
    def calculate_theme_overlap_matrix(self, theme_ids, datasets, limit=50):
        """
        Calculate overlap matrix between sets in different themes
        
        Args:
            theme_ids (list): List of theme IDs
            datasets (dict): Dictionary of pandas DataFrames
            limit (int): Maximum number of sets per theme to include
            
        Returns:
            DataFrame: Overlap matrix
        """
        logger.info(f"Calculating overlap matrix for themes: {theme_ids} with limit {limit}")
        
        # Check if required datasets are available
        required_datasets = ['sets', 'inventories', 'inventory_parts', 'parts']
        missing_datasets = [ds for ds in required_datasets if ds not in datasets]
        if missing_datasets:
            logger.error(f"Missing required datasets for theme overlap matrix: {missing_datasets}")
            st.error(f"Cannot calculate overlap matrix. Missing required datasets: {', '.join(missing_datasets)}")
            return pd.DataFrame()
            
        all_sets = []
        
        # Get sets for each theme
        try:
            for theme_id in theme_ids:
                logger.info(f"Getting sets for theme {theme_id}")
                theme_sets = self.data_processor.get_theme_sets(theme_id, datasets)
                
                if not theme_sets.empty:
                    logger.info(f"Found {len(theme_sets)} sets for theme {theme_id}")
                    # Limit number of sets per theme
                    theme_sets = theme_sets.head(limit)
                    all_sets.append(theme_sets)
                else:
                    logger.warning(f"No sets found for theme {theme_id}")
            
            if not all_sets:
                logger.warning("No sets found for any of the requested themes")
                return pd.DataFrame()
                
        except Exception as e:
            error_details = traceback.format_exc()
            logger.error(f"Error getting theme sets: {str(e)}\n{error_details}")
            st.error(f"Error getting theme sets: {str(e)}")
            return pd.DataFrame()
            
        all_sets_df = pd.concat(all_sets)
        set_nums = all_sets_df['set_num'].tolist()
        
        # Initialize overlap matrix
        matrix = []
        
        # Calculate overlap for each pair of sets
        with st.spinner("Calculating set overlap matrix..."):
            progress_bar = st.progress(0)
            total_pairs = len(set_nums) * (len(set_nums) - 1) // 2
            pair_count = 0
            
            for i, set1 in enumerate(set_nums):
                row = []
                for j, set2 in enumerate(set_nums):
                    if i == j:
                        # Diagonal is 100% overlap (same set)
                        row.append(100)
                    elif j < i:
                        # Lower triangle - copy from upper triangle
                        row.append(matrix[j][i])
                    else:
                        # Upper triangle - calculate overlap
                        overlap = self.calculate_set_overlap(set1, set2, datasets)
                        # Use average of both percentages
                        overlap_pct = (overlap['overlap_percentage1'] + overlap['overlap_percentage2']) / 2
                        row.append(overlap_pct)
                        
                        pair_count += 1
                        progress_bar.progress(pair_count / total_pairs)
                
                matrix.append(row)
            
            progress_bar.empty()
        
        # Convert to DataFrame
        matrix_df = pd.DataFrame(matrix, index=set_nums, columns=set_nums)
        
        return matrix_df
    
    def get_theme_part_counts(self, theme_id, datasets):
        """
        Get part counts for a specific theme
        
        Args:
            theme_id (int): Theme ID
            datasets (dict): Dictionary of pandas DataFrames
            
        Returns:
            dict: Part statistics for the theme
        """
        # Get all sets in this theme
        theme_sets = self.data_processor.get_theme_sets(theme_id, datasets)
        if theme_sets.empty:
            return {
                'total_sets': 0,
                'total_unique_parts': 0,
                'part_frequency': pd.DataFrame()
            }
        
        # Check if inventories dataset is available (may not be when using SQLite/database approach)
        if 'inventories' not in datasets:
            return {
                'total_sets': len(theme_sets),
                'total_unique_parts': 0,
                'part_frequency': pd.DataFrame()
            }
            
        # Get inventory IDs for these sets
        inventory_ids = datasets['inventories'][
            datasets['inventories']['set_num'].isin(theme_sets['set_num'])
        ]['id'].tolist()
        
        if not inventory_ids:
            return {
                'total_sets': len(theme_sets),
                'total_unique_parts': 0,
                'part_frequency': pd.DataFrame()
            }
            
        # Check if inventory_parts dataset is available
        if 'inventory_parts' not in datasets:
            return {
                'total_sets': len(theme_sets),
                'total_unique_parts': 0,
                'part_frequency': pd.DataFrame()
            }
        
        # Get parts for these inventories
        theme_parts = datasets['inventory_parts'][
            datasets['inventory_parts']['inventory_id'].isin(inventory_ids)
        ]
        
        if theme_parts.empty:
            return {
                'total_sets': len(theme_sets),
                'total_unique_parts': 0,
                'part_frequency': pd.DataFrame()
            }
        
        # Count frequency of each part across sets
        part_counts = theme_parts.groupby('part_num').agg({
            'quantity': 'sum',
            'inventory_id': 'nunique'
        }).reset_index()
        
        # Rename columns
        part_counts.rename(columns={
            'quantity': 'total_quantity',
            'inventory_id': 'sets_count'
        }, inplace=True)
        
        # Calculate percentage of sets containing this part
        part_counts['sets_percentage'] = part_counts['sets_count'] / len(inventory_ids) * 100
        
        # Add part information
        if 'parts' in datasets and not datasets['parts'].empty:
            part_counts = part_counts.merge(
                datasets['parts'][['part_num', 'name']],
                on='part_num',
                how='left'
            )
        
        # Sort by frequency
        part_counts = part_counts.sort_values('sets_percentage', ascending=False)
        
        return {
            'total_sets': len(theme_sets),
            'total_unique_parts': len(part_counts),
            'part_frequency': part_counts
        }
