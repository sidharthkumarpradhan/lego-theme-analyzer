import pandas as pd
import numpy as np
import streamlit as st
from scipy.optimize import linear_sum_assignment

class SetOptimizer:
    """
    Optimize selection of LEGO sets to maximize theme coverage
    """
    def __init__(self, data_processor, overlap_analyzer):
        """
        Initialize set optimizer
        
        Args:
            data_processor (DataProcessor): Data processor instance
            overlap_analyzer (OverlapAnalyzer): Overlap analyzer instance
        """
        self.data_processor = data_processor
        self.overlap_analyzer = overlap_analyzer
    
    def find_minimal_sets_for_theme(self, theme_id, datasets, max_sets=5):
        """
        Find the minimal combination of sets that maximize coverage for a theme
        
        Args:
            theme_id (int): Theme ID
            datasets (dict): Dictionary of pandas DataFrames
            max_sets (int): Maximum number of sets to recommend
            
        Returns:
            dict: Optimization results
        """
        # Get theme stats
        theme_stats = self.overlap_analyzer.get_theme_part_counts(theme_id, datasets)
        
        if theme_stats['total_sets'] == 0:
            return {
                'selected_sets': [],
                'coverage': 0,
                'total_parts': 0,
                'unique_parts': 0
            }
        
        # Get all sets in this theme
        theme_sets = self.data_processor.get_theme_sets(theme_id, datasets)
        if theme_sets.empty:
            return {
                'selected_sets': [],
                'coverage': 0,
                'total_parts': 0,
                'unique_parts': 0
            }
        
        # Get inventory IDs for these sets
        inventories = datasets['inventories'][
            datasets['inventories']['set_num'].isin(theme_sets['set_num'])
        ]
        
        # Get parts for each set
        set_parts = {}
        all_parts = set()
        
        with st.spinner("Analyzing parts for theme optimization..."):
            for _, row in theme_sets.iterrows():
                set_num = row['set_num']
                set_name = row['name']
                
                # Get inventory ID for this set
                inventory_ids = inventories[inventories['set_num'] == set_num]['id'].tolist()
                
                if not inventory_ids:
                    continue
                
                # Get parts for this set
                set_inventory_parts = datasets['inventory_parts'][
                    datasets['inventory_parts']['inventory_id'].isin(inventory_ids)
                ]
                
                if set_inventory_parts.empty:
                    continue
                
                # Get unique parts
                parts = set(set_inventory_parts['part_num'].unique())
                set_parts[set_num] = {
                    'name': set_name,
                    'parts': parts,
                    'part_count': len(parts),
                    'year': row.get('year', 0)
                }
                
                # Update all parts
                all_parts.update(parts)
        
        if not set_parts:
            return {
                'selected_sets': [],
                'coverage': 0,
                'total_parts': 0,
                'unique_parts': 0
            }
        
        # Greedy algorithm to find optimal set combination
        selected_sets = []
        covered_parts = set()
        remaining_sets = set_parts.copy()
        
        while len(selected_sets) < max_sets and remaining_sets and covered_parts != all_parts:
            # Find set with most uncovered parts
            best_set = None
            best_new_parts = 0
            
            for set_num, set_info in remaining_sets.items():
                new_parts = len(set_info['parts'] - covered_parts)
                if new_parts > best_new_parts:
                    best_set = set_num
                    best_new_parts = new_parts
            
            if best_set is None or best_new_parts == 0:
                # No more sets add value
                break
            
            # Add this set
            selected_sets.append({
                'set_num': best_set,
                'name': remaining_sets[best_set]['name'],
                'new_parts': best_new_parts,
                'total_parts': remaining_sets[best_set]['part_count'],
                'year': remaining_sets[best_set]['year']
            })
            
            # Update covered parts
            covered_parts.update(remaining_sets[best_set]['parts'])
            
            # Remove from remaining sets
            del remaining_sets[best_set]
        
        # Calculate coverage
        coverage_percentage = len(covered_parts) / len(all_parts) * 100 if all_parts else 0
        
        return {
            'selected_sets': selected_sets,
            'coverage': coverage_percentage,
            'total_parts': len(all_parts),
            'unique_parts': len(covered_parts)
        }
    
    def analyze_buildability(self, available_parts, theme_id, datasets):
        """
        Analyze which sets in a theme can be built with available parts
        
        Args:
            available_parts (dict): Dictionary of available parts {part_num: quantity}
            theme_id (int): Theme ID
            datasets (dict): Dictionary of pandas DataFrames
            
        Returns:
            dict: Buildability analysis
        """
        # Get all sets in this theme
        theme_sets = self.data_processor.get_theme_sets(theme_id, datasets)
        if theme_sets.empty:
            return {
                'buildable_sets': [],
                'missing_parts': {}
            }
        
        buildable_sets = []
        missing_parts_by_set = {}
        
        with st.spinner("Analyzing buildability..."):
            progress_bar = st.progress(0)
            total_sets = len(theme_sets)
            
            for i, (_, row) in enumerate(theme_sets.iterrows()):
                set_num = row['set_num']
                set_name = row['name']
                
                # Get parts for this set
                set_parts = self.data_processor.get_set_parts(set_num, datasets)
                if set_parts.empty:
                    progress_bar.progress((i + 1) / total_sets)
                    continue
                
                # Aggregate parts by part_num
                set_parts_agg = set_parts.groupby('part_num').agg({
                    'quantity': 'sum',
                    'name': 'first',
                    'is_spare': lambda x: 'f' in x.values
                }).reset_index()
                
                # Check if all parts are available in sufficient quantities
                missing_parts = []
                for _, part_row in set_parts_agg.iterrows():
                    part_num = part_row['part_num']
                    required_qty = part_row['quantity']
                    is_spare = part_row['is_spare']
                    available_qty = available_parts.get(part_num, 0)
                    
                    if available_qty < required_qty and not is_spare:
                        missing_parts.append({
                            'part_num': part_num,
                            'name': part_row['name'],
                            'required': required_qty,
                            'available': available_qty,
                            'shortfall': required_qty - available_qty
                        })
                
                # Calculate buildability percentage
                total_required_parts = set_parts_agg['quantity'].sum()
                covered_parts = sum(min(available_parts.get(row['part_num'], 0), row['quantity'])
                                   for _, row in set_parts_agg.iterrows())
                
                buildability = covered_parts / total_required_parts * 100 if total_required_parts > 0 else 0
                
                # Add to buildable sets if all essential parts are available
                if not missing_parts:
                    buildable_sets.append({
                        'set_num': set_num,
                        'name': set_name,
                        'buildability': 100,
                        'year': row.get('year', None),
                        'num_parts': total_required_parts
                    })
                else:
                    # Store missing parts
                    missing_parts_by_set[set_num] = {
                        'set_name': set_name,
                        'buildability': buildability,
                        'missing_parts': missing_parts,
                        'year': row.get('year', None),
                        'num_parts': total_required_parts
                    }
                
                progress_bar.progress((i + 1) / total_sets)
            
            progress_bar.empty()
        
        # Sort buildable sets by number of parts (increasing)
        buildable_sets.sort(key=lambda x: x['num_parts'])
        
        return {
            'buildable_sets': buildable_sets,
            'missing_parts': missing_parts_by_set
        }
