import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import logging
from datetime import datetime
import time
import json
import os
import sqlite3
import traceback
from logger_config import get_logger

# Initialize logger for this module
logger = get_logger(__name__)
from rebrickable_api import RebrickableAPI
from data_processor import DataProcessor
from theme_predictor import ThemePredictor
from overlap_analyzer import OverlapAnalyzer
from set_optimizer import SetOptimizer
from database_manager import DatabaseManager
from utils import (
    create_donut_chart, create_bar_chart, create_heatmap, 
    get_theme_color_map, export_to_csv, format_part_list
)

# Configure page
st.set_page_config(
    page_title="LEGO Theme & Part Analyzer",
    page_icon="logo.svg",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for models and datasets
if 'loaded_data' not in st.session_state:
    st.session_state.loaded_data = False
if 'datasets' not in st.session_state:
    st.session_state.datasets = {}
if 'theme_predictor' not in st.session_state:
    st.session_state.theme_predictor = None
if 'theme_predictor_trained' not in st.session_state:
    st.session_state.theme_predictor_trained = False
if 'user_parts' not in st.session_state:
    # Initialize with empty dictionary, we'll load real parts later
    st.session_state.user_parts = {}
if 'using_sqlite' not in st.session_state:
    st.session_state.using_sqlite = False
if 'show_landing_page' not in st.session_state:
    st.session_state.show_landing_page = True
if 'auto_prediction_results' not in st.session_state:
    st.session_state.auto_prediction_results = None

# Custom CSS with LEGO logo
st.markdown("""
<style>
    .logo-container {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
    }
    .logo-container img {
        height: 50px;
        margin-right: 20px;
    }
    .app-title {
        font-size: 2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# App header with LEGO logo
st.markdown("""
<div class="logo-container">
    <img src="https://assets.lego.com/logos/v4.5.0/brand-lego.svg" alt="LEGO Logo">
    <span class="app-title">LEGO Theme Analyzer & Set Optimizer</span>
</div>
""", unsafe_allow_html=True)

st.markdown("""
A comprehensive tool for LEGO enthusiasts
""")

# Initialize API, database and processors
logger.info("Initializing core components...")
api = RebrickableAPI()
db_manager = DatabaseManager('./lego_data/lego.db')
data_processor = DataProcessor(api)
overlap_analyzer = OverlapAnalyzer(data_processor)
set_optimizer = SetOptimizer(data_processor, overlap_analyzer)
logger.info("Core components initialized")

# Add database manager to session state if not already there
if 'db_manager' not in st.session_state:
    st.session_state.db_manager = db_manager

# Sidebar content
with st.sidebar:
    # Display LEGO logo in sidebar
    st.image("https://assets.lego.com/logos/v4.5.0/brand-lego.svg", width=150)
    
    # Use button only for loading data, no source selection
    st.markdown("### 1️⃣ Load Data")
    st.markdown("**FIRST STEP**: Click below to load LEGO datasets. This must be done before any analysis.")
    
    # Check if we already have a fresh database
    db_exists = db_manager.db_exists()
    db_is_fresh = db_manager.is_data_fresh(max_age_days=7)
    data_source_text = ""
    
    # Data source options - simplified to make the right choice automatically
    if db_exists and db_is_fresh:
        st.info("Using existing database (loaded within the last 7 days)")
        data_source = "Use SQLite Database"
    else:
        st.info("Will download fresh data from Rebrickable.com")
        data_source = "Download from Rebrickable"
    
    if st.button("Load Data"):
        # Use SQLite if selected and available
        if data_source.startswith("Use SQLite Database") and db_exists:
            with st.spinner("Loading data from SQLite database..."):
                try:
                    logger.info("Loading data from SQLite database...")
                    # Load themes from SQLite for a quick check
                    themes_df = db_manager.query_to_dataframe("SELECT * FROM themes")
                    
                    if not themes_df.empty:
                        # Load just basic data from SQLite into memory for operations
                        st.session_state.datasets = {
                            'themes': themes_df,
                            'sets': db_manager.query_to_dataframe("SELECT * FROM sets")
                        }
                        
                        # Add a DB flag to indicate we're using the database
                        st.session_state.using_sqlite = True
                        st.session_state.loaded_data = True
                        data_source_text = "from SQLite database"
                        logger.info(f"Successfully loaded {len(themes_df)} themes and {len(st.session_state.datasets['sets'])} sets from SQLite database")
                        st.success(f"Data loaded successfully from SQLite database.")
                    else:
                        logger.warning("SQLite database exists but appears to be empty")
                        st.warning("SQLite database exists but appears to be empty. Switching to download mode.")
                        # Fall back to download if database exists but is empty
                        db_exists = False
                        db_is_fresh = False
                except Exception as e:
                    error_details = traceback.format_exc()
                    logger.error(f"Error loading data from SQLite: {str(e)}\n{error_details}")
                    st.error(f"Error loading data from SQLite: {str(e)}")
                    st.warning("Could not load from SQLite database. Switching to download mode.")
                    db_exists = False
                    db_is_fresh = False
        
        # If SQLite not selected or failed, download from Rebrickable
        if not db_exists or not db_is_fresh or data_source.startswith("Download from Rebrickable"):
            # Try to load real data from Rebrickable
            with st.spinner("Downloading data from Rebrickable..."):
                try:
                    logger.info("Starting download from Rebrickable.com...")
                    # Use the API to download datasets directly to SQLite (memory efficient)
                    result = api.download_datasets(cache=True, use_db=True)
                    
                    # Check if we got the data
                    if result and isinstance(result, dict) and 'db_manager' in result:
                        logger.info("Successfully downloaded and processed Rebrickable data")
                        st.session_state.loaded_data = True
                        data_source_text = "from Rebrickable"
                        
                        # Update database manager reference 
                        st.session_state.db_manager = result['db_manager']
                        db_manager = result['db_manager']
                        
                        # Store minimal metadata instead of full datasets
                        st.session_state.dataset_metadata = result.get('metadata', {})
                        logger.info(f"Dataset metadata: {result.get('metadata', {})}")
                        
                        # Flag that we're using SQLite
                        st.session_state.using_sqlite = True
                        
                        # Load essential datasets (small ones only) for UI components
                        st.session_state.datasets = {}
                        for name in ['themes', 'sets']:
                            try:
                                if db_manager:
                                    logger.info(f"Loading {name} dataset into memory")
                                    # Load only these essential datasets in memory (they're small)
                                    st.session_state.datasets[name] = db_manager.query_to_dataframe(f"SELECT * FROM {name}")
                                    logger.info(f"Loaded {len(st.session_state.datasets[name])} {name} into memory")
                            except Exception as e:
                                error_details = traceback.format_exc()
                                logger.error(f"Could not load {name} dataset from database: {str(e)}\n{error_details}")
                                st.warning(f"Could not load {name} dataset from database: {str(e)}")
                        
                        st.success(f"Data loaded successfully! Using SQLite for efficient memory usage.")
                    else:
                        logger.error(f"Failed to download data from Rebrickable. Result: {result}")
                        st.error("Failed to download data from Rebrickable. Please try again later.")
                        st.session_state.loaded_data = False
                        st.session_state.using_sqlite = False
                except Exception as e:
                    error_details = traceback.format_exc()
                    logger.error(f"Error downloading data: {str(e)}\n{error_details}")
                    st.error(f"Error downloading data: {str(e)}")
                    st.session_state.loaded_data = False
                    st.session_state.using_sqlite = False
                
            # Initialize theme predictor if data loaded successfully
            if st.session_state.loaded_data:
                with st.spinner("Initializing theme predictor..."):
                    try:
                        logger.info("Preparing data for theme predictor training...")
                        # Prepare data for theme prediction using database for memory efficiency
                        db = st.session_state.db_manager if st.session_state.using_sqlite else None
                        
                        # Ensure required datasets for theme prediction
                        required_datasets = ['sets', 'inventories', 'inventory_parts', 'themes']
                        st.session_state.datasets = data_processor.ensure_datasets_loaded(
                            required_datasets, 
                            st.session_state.datasets, 
                            db
                        )
                        
                        X, y = data_processor.prepare_data_for_theme_prediction(
                            datasets=st.session_state.datasets, 
                            db_manager=db
                        )
                        
                        if X is not None and y is not None and X.shape[0] > 0 and y.shape[0] > 0:
                            logger.info(f"Data prepared: X shape: {X.shape}, y shape: {y.shape}")
                            # Get theme names
                            theme_map = data_processor.get_theme_name_mapping(st.session_state.datasets)
                            logger.info(f"Theme name mapping created with {len(theme_map)} themes")
                            
                            # Initialize and train model with reduced memory footprint
                            logger.info("Initializing and training theme predictor model...")
                            st.session_state.theme_predictor = ThemePredictor()
                            success = st.session_state.theme_predictor.train(X, y, theme_map, memory_efficient=True)
                            
                            if success:
                                logger.info("Theme predictor trained successfully")
                                st.session_state.theme_predictor_trained = True
                                
                                # Successfully initialized the theme predictor
                                st.success("Theme predictor initialized successfully!")
                                st.info("You can now analyze real LEGO sets or your own collection.")
                                
                                # Explicitly log the inventory before prediction
                                logger.info(f"User parts before prediction: {st.session_state.user_parts}")
                                
                                # Automatically run prediction with user inventory
                                try:
                                    logger.info(f"Automatically predicting themes for sample inventory with {len(st.session_state.user_parts)} parts")
                                    
                                    # Get prediction
                                    prediction = st.session_state.theme_predictor.predict(st.session_state.user_parts)
                                    logger.info(f"Prediction result: {prediction}")
                                    
                                    # Store prediction in session state
                                    if prediction and 'top_themes' in prediction:
                                        logger.info("Setting auto_prediction_results in session state")
                                        st.session_state.auto_prediction_results = prediction
                                        logger.info(f"Auto-prediction successful with top theme: {prediction['theme_name']}")
                                        st.info(f"Your inventory matches best with the '{prediction['theme_name']}' theme. Check the Theme Prediction tab for details.")
                                        
                                        # Force a rerun to ensure UI updates
                                        st.rerun()
                                    else:
                                        logger.warning("Auto-prediction returned no results")
                                        st.warning("Automatic theme prediction couldn't determine a theme. Try adding more parts to your inventory.")
                                except Exception as e:
                                    error_details = traceback.format_exc()
                                    logger.error(f"Error during automatic theme prediction: {str(e)}\n{error_details}")
                                    st.error(f"Error during theme prediction: {str(e)}")
                                    # Show error to help debugging
                            else:
                                logger.warning("Theme predictor training failed")
                                st.session_state.theme_predictor_trained = False
                                st.warning("Could not initialize theme predictor. Some theme prediction features will be limited.")
                        else:
                            logger.warning("Could not prepare data for theme prediction. X or y is None or empty.")
                            st.session_state.theme_predictor_trained = False
                            st.warning("Could not prepare sufficient data for theme prediction. Some features may be limited.")
                    except Exception as e:
                        error_details = traceback.format_exc()
                        logger.error(f"Error initializing theme predictor: {str(e)}\n{error_details}")
                        st.error(f"Error initializing theme predictor: {str(e)}")
                        st.session_state.theme_predictor = None
                        st.session_state.theme_predictor_trained = False
    
    st.markdown("---")
    st.header("2️⃣ Enter Your LEGO Inventory")
    
    # User inventory management - simplified to focus on real data
    upload_option = st.selectbox(
        "Manage Inventory",
        ["Edit Manually", "Import from CSV"]
    )
    
    if upload_option == "Import from CSV":
        # Add icon with example CSV format
        col1, col2 = st.columns([5, 1])
        col1.write("Import your LEGO inventory from a CSV file with 'part_num' and 'quantity' columns.")
        
        # Add example and help button
        with col2:
            if st.button("ℹ️ Example CSV"):
                st.info("""
                Your CSV file should look like this:
                ```
                part_num,quantity
                3001,5
                3002,10
                3003,2
                ```
                You can download a sample template below:
                """)
                # Create a simple CSV example
                sample_csv = """part_num,quantity
3001,5
3002,10
3003,2
3622,4
3710,8
2412,12
"""
                # Provide download link for sample CSV
                st.download_button(
                    label="Download Sample CSV",
                    data=sample_csv,
                    file_name="lego_inventory_template.csv",
                    mime="text/csv"
                )
        
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file is not None:
            try:
                inventory_df = pd.read_csv(uploaded_file)
                if 'part_num' in inventory_df.columns and 'quantity' in inventory_df.columns:
                    uploaded_inventory = {
                        row['part_num']: row['quantity'] for _, row in inventory_df.iterrows()
                    }
                    st.session_state.user_parts = uploaded_inventory
                    st.success(f"Imported {len(st.session_state.user_parts)} parts successfully!")
                else:
                    st.error("CSV must have 'part_num' and 'quantity' columns")
            except Exception as e:
                st.error(f"Error importing CSV: {e}")
    
    elif upload_option == "Edit Manually":
        st.write("Add LEGO parts to your inventory by entering the part number and quantity.")
        
        # Create a form for adding new parts
        with st.form("add_part_form"):
            col1, col2 = st.columns([3, 1])
            new_part_num = col1.text_input("Part Number", placeholder="Enter LEGO part number (e.g., 3001)")
            new_part_qty = col2.number_input("Quantity", min_value=1, value=1)
            submit_button = st.form_submit_button("Add Part")
            
            if submit_button and new_part_num.strip():
                # Add the new part to the inventory
                if new_part_num in st.session_state.user_parts:
                    # If part already exists, update quantity
                    st.session_state.user_parts[new_part_num] += new_part_qty
                    st.info(f"Updated quantity for part {new_part_num}")
                else:
                    # Add new part
                    st.session_state.user_parts[new_part_num] = new_part_qty
                    st.success(f"Added part {new_part_num} (Qty: {new_part_qty})")
        
        # Display current inventory with edit options in a table
        if st.session_state.user_parts:
            st.write("### Your Current Inventory")
            
            # Create dataframe from inventory for display
            inventory_data = []
            for part_num, quantity in sorted(st.session_state.user_parts.items()):
                if part_num.strip():  # Skip empty part numbers
                    inventory_data.append({
                        "Part Number": part_num,
                        "Quantity": quantity
                    })
            
            if inventory_data:
                inventory_df = pd.DataFrame(inventory_data)
                
                # Display the inventory as a dataframe with edit options
                edited_df = st.data_editor(
                    inventory_df,
                    key="inventory_editor",
                    num_rows="dynamic",
                    column_config={
                        "Part Number": st.column_config.TextColumn(
                            "Part Number",
                            help="LEGO part number",
                            width="medium"
                        ),
                        "Quantity": st.column_config.NumberColumn(
                            "Quantity",
                            help="Number of this part in your inventory",
                            min_value=0,
                            step=1,
                            width="small"
                        )
                    },
                    hide_index=True
                )
                
                # Update inventory based on the edited dataframe
                if edited_df is not None and not edited_df.empty:
                    # Create a new dictionary from the edited dataframe
                    updated_inventory = {}
                    for _, row in edited_df.iterrows():
                        part_num = str(row["Part Number"]).strip()
                        quantity = int(row["Quantity"])
                        if part_num and quantity > 0:
                            updated_inventory[part_num] = quantity
                    
                                    # Update the session state with the new inventory
                    st.session_state.user_parts = updated_inventory
        
        if st.button("Clear All Parts"):
            st.session_state.user_parts = {}
            st.success("Inventory cleared!")
    
    st.info(f"Current inventory: {len(st.session_state.user_parts)} unique parts")
    
    # Add a prominent analyze button that predicts themes based on current inventory
    st.markdown("---")
    st.header("3️⃣ Analyze Your Inventory")
    
    if not st.session_state.loaded_data:
        st.warning("⚠️ Please click the 'Load Data' button above to download LEGO datasets first.")
        
        if st.button("🔍 Analyze Inventory", type="primary", disabled=True):
            pass
            
        st.info("You need to load data before analyzing your inventory. Click 'Load Data' in the sidebar first.")
    else:
        # Show inventory size for clarity
        inventory_size = len(st.session_state.user_parts) if hasattr(st.session_state, 'user_parts') else 0
        
        if inventory_size == 0:
            st.warning("Your inventory is empty. Please add parts in the inventory section above before analyzing.")
        else:
            st.write(f"You have {inventory_size} unique parts in your inventory. Click the button below to predict which LEGO themes your parts belong to:")
        
        if st.button("🔍 Analyze Inventory", type="primary", disabled=(inventory_size == 0)):
            if not st.session_state.user_parts:
                st.error("Please add parts to your inventory first.")
            else:
                with st.spinner("Analyzing your inventory..."):
                    try:
                        # First try to initialize theme predictor if it doesn't exist
                        if not hasattr(st.session_state, 'theme_predictor') or st.session_state.theme_predictor is None:
                            st.info("Initializing theme predictor...")
                            try:
                                logger.info("Attempting to initialize theme predictor...")
                                # Prepare data for theme prediction using database for memory efficiency
                                db = st.session_state.db_manager if hasattr(st.session_state, 'using_sqlite') and st.session_state.using_sqlite else None
                                X, y = data_processor.prepare_data_for_theme_prediction(
                                    datasets=st.session_state.datasets, 
                                    db_manager=db
                                )
                                
                                if X is not None and y is not None:
                                    logger.info(f"Data prepared: X shape: {X.shape}, y shape: {y.shape}")
                                    # Get theme names
                                    theme_map = data_processor.get_theme_name_mapping(st.session_state.datasets)
                                    logger.info(f"Theme name mapping created with {len(theme_map)} themes")
                                    
                                    # Initialize and train model with reduced memory footprint
                                    logger.info("Initializing and training theme predictor model...")
                                    st.session_state.theme_predictor = ThemePredictor()
                                    st.session_state.theme_predictor.train(X, y, theme_map, memory_efficient=True)
                                    logger.info("Theme predictor trained successfully")
                                    st.session_state.theme_predictor_trained = True
                                    st.success("Theme predictor initialized successfully!")
                                else:
                                    st.error("Could not prepare data for theme prediction. Please reload the data.")
                            except Exception as e:
                                error_details = traceback.format_exc()
                                logger.error(f"Error initializing theme predictor: {str(e)}\n{error_details}")
                                st.error(f"Error initializing theme predictor: {str(e)}")
                        
                        # Now run prediction with initialized predictor
                        if hasattr(st.session_state, 'theme_predictor') and st.session_state.theme_predictor is not None:
                            # Run prediction on current inventory
                            prediction = st.session_state.theme_predictor.predict(st.session_state.user_parts)
                            
                            if prediction:
                                # Store results for display on the Theme Prediction tab
                                st.session_state.auto_prediction_results = prediction
                                
                                # Display a preview of the results
                                st.success(f"Analysis complete! Top theme: **{prediction['theme_name']}**")
                                
                                # Show top 3 themes with confidence
                                st.subheader("🔍 Theme Analysis Results")
                                theme_df = pd.DataFrame(prediction['top_themes'][:3])
                                theme_df['confidence'] = theme_df['confidence'] * 100
                                
                                # Display as a table
                                st.dataframe(
                                    theme_df.rename(columns={
                                        'theme_name': 'Theme',
                                        'confidence': 'Match Confidence (%)'
                                    }).style.format({
                                        'Match Confidence (%)': '{:.1f}%'
                                    })
                                )
                                
                                # Prompt to go to the Theme Prediction tab for more details
                                st.info("👉 For full analysis details, switch to the 'Theme Prediction' tab")
                        else:
                            st.error("Theme predictor could not be initialized. Please reload the data.")
                    except Exception as e:
                        st.error(f"Could not analyze inventory: {e}")

# Check if lego_data directory exists
if not os.path.exists('./lego_data'):
    os.makedirs('./lego_data', exist_ok=True)

# Custom CSS for the footer and banner - moved outside of conditional block so it's always available
st.markdown("""
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #f5f5f5;
    color: #666;
    text-align: center;
    padding: 10px;
    font-size: 14px;
    border-top: 1px solid #ddd;
    z-index: 100;  /* Higher z-index to appear above other elements */
}
.footer p {
    margin: 0;
    padding: 2px;
}
.heart {
    color: red;
}
.banner {
    background-color: #FFEB3B;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 60px;  /* Add margin at bottom to prevent overlap with footer */
    text-align: center;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    z-index: 1;  /* Lower z-index to appear below footer */
}
</style>
""", unsafe_allow_html=True)

# Main content area - changes based on data loading state
if not st.session_state.loaded_data:
    # Display landing page when data is not loaded
    st.markdown("## Welcome to LEGO Theme Analyzer & Set Optimizer")
    
    # Create three columns for feature highlights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🔍 Theme Prediction")
        st.image("https://cdn-icons-png.flaticon.com/512/4341/4341061.png", width=100)
        st.markdown("""
        * Predict LEGO themes from parts
        * Upload your inventory or enter manually
        * See confidence scores for predictions
        """)
    
    with col2:
        st.markdown("### 📊 Part Analysis")
        st.image("https://cdn-icons-png.flaticon.com/512/8332/8332651.png", width=100)
        st.markdown("""
        * Analyze part overlap between sets
        * Find common parts across themes
        * Identify unique parts for each theme
        """)
    
    with col3:
        st.markdown("### 🛒 Set Optimizer")
        st.image("https://cdn-icons-png.flaticon.com/512/3837/3837365.png", width=100)
        st.markdown("""
        * Find optimal sets to purchase
        * Maximize buildable sets per theme
        * Analyze your inventory vs. themes
        """)
    
    # Add a call to action
    st.markdown("---")
    st.markdown("### 🚀 Get Started")
    st.markdown("""
    1. Click the **Load Data** button in the sidebar to download LEGO data
    2. Wait for the automated theme prediction model to train
    3. Start exploring and analyzing your LEGO collection!
    """)
    
    # Add some tips
    with st.expander("Tips for best results"):
        st.markdown("""
        * For best performance, use the database mode (default)
        * Training the theme predictor may take a few minutes initially
        * You can add parts to your inventory manually or via CSV upload
        * CSV files should have 'part_num' and 'quantity' columns
        """)
    
    # Add About this Application section
    st.markdown("---")
    st.markdown("### About this Application")
    st.markdown("""
    This application was developed to help LEGO enthusiasts analyze their collections, 
    predict set themes based on part inventories, and optimize purchasing decisions.
    
    The app uses machine learning algorithms to analyze patterns in LEGO sets and provide
    insights about part usage across different themes.
    """)
    
    # Add credits and copyright info
    st.markdown("---")
    
else:
    # Create tabs for different analyses when data is loaded
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Theme Prediction", 
        "Part Overlap Analysis", 
        "Set Purchase Optimizer",
        "Theme Requirements",
        "Data Browser",
        "SQL Query"
    ])
    
    # Initialize variables to avoid errors
    theme_options = []
    theme_map = {}
    
    # Set current tab selection for conditional messages
    tab_selection = "Theme Prediction"
    
    # Tab 1: Theme Prediction
    with tab1:
        st.header("Theme Prediction")
        st.markdown("""
        **What is this?** This tab uses machine learning to predict LEGO themes based on YOUR brick collection.
        
        **Why is this useful?** If you have a collection of LEGO parts but don't know what theme they belong to, 
        this tool can analyze the part frequencies and tell you the most likely theme.
        
        **How to use:** Add YOUR LEGO parts in the inventory section below, then come back to this tab.
        Results will automatically update as you edit your inventory!
        """)

        # Debug info hidden in production but logged for troubleshooting
        # Log status to help with debugging
        if hasattr(st.session_state, 'loaded_data'):
            logger.info(f"Data loaded status: {st.session_state.loaded_data}")
        
        if hasattr(st.session_state, 'theme_predictor'):
            logger.info(f"Theme predictor initialized: {st.session_state.theme_predictor is not None}")
            
        if hasattr(st.session_state, 'auto_prediction_results'):
            logger.info(f"Automatic prediction results available: {st.session_state.auto_prediction_results is not None}")
        else:
            logger.info("No automatic prediction results available")

        # Check if data is loaded but predictor isn't initialized
        if hasattr(st.session_state, 'loaded_data') and st.session_state.loaded_data:
            if not hasattr(st.session_state, 'theme_predictor') or st.session_state.theme_predictor is None:
                # Add a button to initialize the predictor manually
                st.warning("Theme predictor not initialized yet.")
                
                if st.button("⚙️ Initialize Theme Predictor", key="init_predictor_btn", type="primary"):
                    with st.spinner("Initializing theme predictor (this may take a minute)..."):
                        try:
                            # Prepare data for theme prediction using database for memory efficiency
                            db = st.session_state.db_manager if hasattr(st.session_state, 'using_sqlite') and st.session_state.using_sqlite else None
                            X, y = data_processor.prepare_data_for_theme_prediction(
                                datasets=st.session_state.datasets, 
                                db_manager=db
                            )
                            
                            if X is not None and y is not None:
                                logger.info(f"Data prepared: X shape: {X.shape}, y shape: {y.shape}")
                                # Get theme names
                                theme_map = data_processor.get_theme_name_mapping(st.session_state.datasets)
                                logger.info(f"Theme name mapping created with {len(theme_map)} themes")
                                
                                # Initialize and train model with reduced memory footprint
                                logger.info("Initializing and training theme predictor model...")
                                st.session_state.theme_predictor = ThemePredictor()
                                success = st.session_state.theme_predictor.train(X, y, theme_map, memory_efficient=True)
                                
                                if success:
                                    logger.info("Theme predictor trained successfully")
                                    st.session_state.theme_predictor_trained = True
                                    st.success("Theme predictor initialized successfully!")
                                    st.rerun()  # Force a refresh
                                else:
                                    st.error("Failed to train theme predictor.")
                            else:
                                st.error("Could not prepare data for theme prediction.")
                        except Exception as e:
                            error_details = traceback.format_exc()
                            logger.error(f"Error initializing theme predictor: {str(e)}\n{error_details}")
                            st.error(f"Error initializing theme predictor: {str(e)}")
                
                # Show message that user needs to initialize predictor
                st.info("Please initialize the theme predictor to continue.")
            
            # If we have the predictor but no results, add a dedicated predict button
            elif not hasattr(st.session_state, 'auto_prediction_results') or st.session_state.auto_prediction_results is None:
                
                # Add option to load real set parts from database
                st.subheader("Select a Real LEGO Set for Analysis")
                st.write("Instead of sample parts, you can analyze a real LEGO set from our database:")
                
                # Get a list of popular sets to choose from
                if st.session_state.using_sqlite and hasattr(st.session_state, 'db_manager'):
                    try:
                        popular_sets_query = """
                        SELECT sets.set_num, sets.name, sets.year, sets.num_parts, themes.name as theme_name 
                        FROM sets 
                        JOIN themes ON sets.theme_id = themes.id
                        WHERE sets.num_parts > 50
                        ORDER BY sets.num_parts DESC
                        LIMIT 100
                        """
                        popular_sets = st.session_state.db_manager.query_to_dataframe(popular_sets_query)
                        
                        if not popular_sets.empty:
                            # Create a format function for the selectbox
                            def format_set_option(row_idx):
                                row = popular_sets.iloc[row_idx]
                                return f"{row['set_num']}: {row['name']} ({row['year']}) - {row['num_parts']} parts - {row['theme_name']}"
                            
                            set_options = list(range(len(popular_sets)))
                            selected_set_idx = st.selectbox(
                                "Choose a LEGO set to analyze:",
                                options=set_options,
                                format_func=format_set_option
                            )
                            
                            selected_set = popular_sets.iloc[selected_set_idx]
                            st.write(f"Selected: **{selected_set['name']}** ({selected_set['num_parts']} parts)")
                            
                            if st.button("🧱 Load Set Parts & Predict Theme", type="primary"):
                                with st.spinner(f"Loading parts for set {selected_set['set_num']}..."):
                                    # Get parts for this set from the database
                                    set_parts = data_processor.get_set_parts(selected_set['set_num'], st.session_state.datasets)
                                    
                                    if not set_parts.empty:
                                        # Create parts dictionary in the format needed for prediction
                                        parts_dict = {}
                                        for _, row in set_parts.iterrows():
                                            parts_dict[row['part_num']] = int(row['quantity'])
                                        
                                        # Update the user's inventory with these parts
                                        st.session_state.user_parts = parts_dict
                                        
                                        # Run prediction
                                        st.success(f"Loaded {len(parts_dict)} parts from set {selected_set['set_num']}")
                                        
                                        # Make prediction
                                        try:
                                            prediction = st.session_state.theme_predictor.predict(parts_dict)
                                            if prediction and 'top_themes' in prediction:
                                                st.session_state.auto_prediction_results = prediction
                                                
                                                # Compare with actual theme
                                                actual_theme = selected_set['theme_name']
                                                predicted_theme = prediction['theme_name']
                                                
                                                st.success(f"Prediction successful: {predicted_theme}")
                                                st.info(f"Actual theme: {actual_theme}")
                                                
                                                # Check if prediction was correct
                                                if actual_theme.lower() in predicted_theme.lower() or predicted_theme.lower() in actual_theme.lower():
                                                    st.success("✓ Prediction matches actual theme!")
                                                else:
                                                    st.warning("⚠️ Prediction differs from actual theme")
                                                
                                                st.rerun()  # Force a refresh to show results
                                            else:
                                                st.warning("Could not generate prediction.")
                                        except Exception as e:
                                            st.error(f"Error during prediction: {str(e)}")
                                    else:
                                        st.error(f"Could not find parts for set {selected_set['set_num']}")
                        else:
                            st.warning("No sets found in the database.")
                    except Exception as e:
                        st.error(f"Error loading sets from database: {str(e)}")
                else:
                    st.warning("Database not available. Please load data first.")
                
                # Also add option to use current inventory
                st.subheader("Or Use Your Current Inventory")
                
                inventory_size = len(st.session_state.user_parts) if hasattr(st.session_state, 'user_parts') and st.session_state.user_parts else 0
                
                if inventory_size > 0:
                    st.info(f"You have {inventory_size} parts in your inventory. Click the button below to predict the theme.")
                    
                    if st.button("🔍 Predict Theme from Your Inventory", key="predict_now_btn", type="primary"):
                        with st.spinner("Analyzing your inventory..."):
                            try:
                                prediction = st.session_state.theme_predictor.predict(st.session_state.user_parts)
                                if prediction and 'top_themes' in prediction:
                                    st.session_state.auto_prediction_results = prediction
                                    st.success(f"Prediction successful: {prediction['theme_name']}")
                                    st.rerun()  # Force a refresh to show results
                                else:
                                    st.warning("Could not generate prediction.")
                            except Exception as e:
                                st.error(f"Error during prediction: {str(e)}")
                else:
                    st.warning("No parts in inventory. Please add parts in the inventory section below or select a set above.")
            
            # Check if we have prediction results to display
            if hasattr(st.session_state, 'auto_prediction_results') and st.session_state.auto_prediction_results:
                with st.expander("Recent Automatic Prediction Results", expanded=True):
                    st.markdown("### Theme Prediction Based on Your Inventory")
                    prediction = st.session_state.auto_prediction_results
                    
                    # Top prediction
                    st.metric(
                        label="Predicted Theme", 
                        value=prediction['theme_name'],
                        delta=f"{prediction['confidence']:.1%} confident"
                    )
                    
                    # Top themes
                    theme_df = pd.DataFrame(prediction['top_themes'])
                    theme_df['confidence'] = theme_df['confidence'] * 100
                    
                    # Display as a table
                    st.dataframe(
                        theme_df.rename(columns={
                            'theme_name': 'Theme',
                            'confidence': 'Match Confidence (%)'
                        }).style.format({
                            'Match Confidence (%)': '{:.1f}%'
                        })
                    )
                    
                    st.markdown("---")
        
        st.subheader("Predict Theme from Parts")
        
        prediction_option = st.radio(
            "Prediction Source",
            ["Use Current Inventory", "Import from CSV"]
        )
        
        parts_to_predict = {}
        
        if prediction_option == "Use Current Inventory":
            parts_to_predict = st.session_state.user_parts
            if parts_to_predict and len(parts_to_predict) > 0:
                st.info(f"Using inventory with {len(parts_to_predict)} parts for prediction")
            else:
                st.warning("Your inventory is empty. Please add parts in the inventory section first.")
                st.info("Scroll down to '2️⃣ Enter Your LEGO Inventory' section and add parts.")
            
        elif prediction_option == "Import from CSV":
            uploaded_file = st.file_uploader("Upload CSV file for prediction", type=["csv"], key="prediction_csv")
            if uploaded_file is not None:
                try:
                    prediction_df = pd.read_csv(uploaded_file)
                    if 'part_num' in prediction_df.columns and 'quantity' in prediction_df.columns:
                        parts_to_predict = {
                            row['part_num']: row['quantity'] for _, row in prediction_df.iterrows()
                        }
                        st.success(f"Imported {len(parts_to_predict)} parts for prediction")
                    else:
                        st.error("CSV must have 'part_num' and 'quantity' columns")
                except Exception as e:
                    st.error(f"Error importing CSV: {e}")
        
        if parts_to_predict and st.button("Predict Theme"):
            with st.spinner("Predicting theme..."):
                prediction = st.session_state.theme_predictor.predict(parts_to_predict)
                
                if prediction:
                    # Display prediction results
                    st.subheader("Prediction Results")
                    
                    # Top prediction
                    st.metric(
                        label="Predicted Theme", 
                        value=prediction['theme_name'],
                        delta=f"{prediction['confidence']:.1%} confident"
                    )
                    
                    # Top 3 predictions
                    predictions_df = pd.DataFrame(prediction['top_themes'])
                    predictions_df['confidence'] = predictions_df['confidence'] * 100
                    
                    # Create confidence chart
                    fig = px.bar(
                        predictions_df,
                        x='theme_name',
                        y='confidence',
                        title="Top Theme Predictions",
                        color='confidence',
                        color_continuous_scale='blues',
                        text='confidence'
                    )
                    
                    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show feature importance if available
                    feature_importance = st.session_state.theme_predictor.get_feature_importance()
                    if not feature_importance.empty:
                        st.subheader("Feature Importance")
                        st.markdown("These principal components represent the most important patterns in the parts data for predicting themes:")
                        
                        # Plot feature importance with part information
                        if 'KeyParts' in feature_importance.columns and not feature_importance['KeyParts'].isnull().all():
                            # Create a better display format combining PC and key parts
                            # First, create clean data for display
                            display_df = feature_importance.head(6).copy()  # Limit to 6 for better readability
                            
                            # Create custom labels for x-axis
                            display_df['Label'] = display_df['PC']
                            
                            # Create a table to display part numbers separately
                            st.markdown("### Key Parts Influencing Theme Prediction")
                            part_info_df = display_df[['PC', 'KeyParts', 'Importance']]
                            part_info_df['Importance'] = (part_info_df['Importance'] * 100).round(2).astype(str) + '%'
                            part_info_df.columns = ['Component', 'Most Important LEGO Parts', 'Importance']
                            st.table(part_info_df)
                            
                            # Use the PC only for the chart for cleaner display
                            fig = px.bar(
                                display_df, 
                                x='PC', 
                                y='Importance',
                                title="Component Importance in Theme Prediction",
                                color='Importance',
                                color_continuous_scale='viridis'
                            )
                            fig.update_layout(xaxis_title="Principal Components")
                        else:
                            # Fallback to basic display
                            fig = px.bar(
                                feature_importance.head(10), 
                                x='PC', 
                                y='Importance',
                                title="Top Principal Components for Theme Prediction",
                                color='Importance',
                                color_continuous_scale='viridis'
                            )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Failed to predict theme. Please ensure your inventory contains valid parts.")
        elif not st.session_state.loaded_data:
            # Only show when data is not loaded AND we're on the Theme Prediction tab
            if tab_selection == "Theme Prediction":
                st.info("Please load data first using the sidebar 'Load Data' button to enable theme prediction.")

    # Tab 2: Part Overlap Analysis
    with tab2:
        st.header("Part Overlap Analysis")
        # Update tab selection
        tab_selection = "Part Overlap Analysis"
        
        st.markdown("""
        **What is this?** This tab analyzes how LEGO parts are shared between different sets within a theme or across themes.
        
        **Why is this useful?** Understanding part overlap helps you:
        - Identify which sets share the most common parts
        - Determine which set combinations give you the most diverse parts
        - See if buying one set would help you build another set
        
        **How to use:** Select a theme, adjust the maximum number of sets to compare, and click "Calculate Overlap Matrix" 
        to see a heatmap showing part commonality between sets.
        """)
    
        if st.session_state.loaded_data:
            # Theme selection
            theme_map = data_processor.get_theme_name_mapping(st.session_state.datasets)
            theme_options = [(theme_id, name) for theme_id, name in theme_map.items()]
            theme_options.sort(key=lambda x: x[1])  # Sort by theme name
            
            st.subheader("Compare Sets Within Themes")
        
        col1, col2 = st.columns(2)
        
        selected_theme_id = col1.selectbox(
            "Select Theme",
            options=[t[0] for t in theme_options],
            format_func=lambda x: next((t[1] for t in theme_options if t[0] == x), "Unknown"),
            key="overlap_theme"
        )
        
        max_sets = col2.slider(
            "Maximum Sets to Compare",
            min_value=5,
            max_value=50,
            value=15,
            step=5,
            key="overlap_max_sets"
        )
        
        if st.button("Calculate Overlap Matrix"):
            # First check that we have all necessary datasets or try to load them
            required_datasets = ['sets', 'inventories', 'inventory_parts', 'parts']
            
            # Let the data processor try to load any missing datasets
            with st.spinner("Checking required datasets..."):
                st.session_state.datasets = data_processor.ensure_datasets_loaded(
                    required_datasets, 
                    st.session_state.datasets, 
                    st.session_state.db_manager if st.session_state.using_sqlite else None
                )
            
            # Now check if any are still missing
            missing_datasets = [ds for ds in required_datasets if ds not in st.session_state.datasets]
            
            if missing_datasets:
                st.error(f"Cannot calculate overlap matrix. Missing required datasets: {', '.join(missing_datasets)}")
                st.info("Please reload data from the sidebar and select 'Download from Rebrickable' to get all necessary datasets.")
                logger.error(f"Missing datasets when calculating overlap matrix: {missing_datasets}")
            else:
                # Get sets for selected theme
                theme_sets = data_processor.get_theme_sets(selected_theme_id, st.session_state.datasets)
                
                if theme_sets.empty:
                    st.warning(f"No sets found for the selected theme (ID: {selected_theme_id})")
                    logger.warning(f"No sets found for theme {selected_theme_id} when calculating overlap matrix")
                else:
                    logger.info(f"Calculating overlap matrix for theme {selected_theme_id} with {len(theme_sets)} sets")
                    # Calculate overlap matrix
                    with st.spinner("Calculating overlap matrix (this may take a while)..."):
                        try:
                            overlap_matrix = overlap_analyzer.calculate_theme_overlap_matrix(
                                [selected_theme_id], 
                                st.session_state.datasets,
                                limit=max_sets
                            )
                        except Exception as e:
                            error_details = traceback.format_exc()
                            st.error(f"Error calculating overlap matrix: {str(e)}")
                            logger.error(f"Error calculating overlap matrix: {str(e)}\n{error_details}")
                            overlap_matrix = pd.DataFrame()
                    
                    if overlap_matrix.empty:
                        st.warning("Could not calculate overlap matrix")
                    else:
                        # Get set names for better display
                        set_names = {}
                        for set_num in overlap_matrix.index:
                            set_info = theme_sets[theme_sets['set_num'] == set_num]
                            if not set_info.empty:
                                set_names[set_num] = f"{set_num}: {set_info.iloc[0]['name']}"
                            else:
                                set_names[set_num] = set_num
                        
                        # Rename indices and columns
                        overlap_matrix.index = [set_names.get(idx, idx) for idx in overlap_matrix.index]
                        overlap_matrix.columns = [set_names.get(col, col) for col in overlap_matrix.columns]
                        
                        # Display heatmap
                        st.subheader(f"Part Overlap Matrix - {theme_map.get(selected_theme_id, f'Theme {selected_theme_id}')}")
                        st.markdown("Color represents the percentage of parts that are shared between sets (darker = more overlap)")
                        
                        fig = create_heatmap(
                            overlap_matrix, 
                            f"Part Overlap Between Sets in {theme_map.get(selected_theme_id, f'Theme {selected_theme_id}')}"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display table for detailed view
                        with st.expander("View Overlap Matrix Table"):
                            st.dataframe(overlap_matrix)
                            
                            # Export option
                            st.markdown(export_to_csv(overlap_matrix.reset_index(), f"overlap_matrix_{selected_theme_id}.csv"), unsafe_allow_html=True)
                            
        st.subheader("Compare Sets Between Themes")
        
        col1, col2, col3 = st.columns(3)
        
        selected_theme1 = col1.selectbox(
            "Theme 1",
            options=[t[0] for t in theme_options],
            format_func=lambda x: next((t[1] for t in theme_options if t[0] == x), "Unknown"),
            key="cross_theme1"
        )
        
        selected_theme2 = col2.selectbox(
            "Theme 2",
            options=[t[0] for t in theme_options],
            format_func=lambda x: next((t[1] for t in theme_options if t[0] == x), "Unknown"),
            key="cross_theme2",
            index=min(1, len(theme_options) - 1)  # Select second theme by default
        )
        
        cross_max_sets = col3.slider(
            "Max Sets per Theme",
            min_value=5,
            max_value=25,
            value=10,
            step=5,
            key="cross_max_sets"
        )
        
        if st.button("Calculate Cross-Theme Overlap"):
            # First check that we have all necessary datasets or try to load them
            required_datasets = ['sets', 'inventories', 'inventory_parts', 'parts']
            
            # Let the data processor try to load any missing datasets
            with st.spinner("Checking required datasets for cross-theme analysis..."):
                st.session_state.datasets = data_processor.ensure_datasets_loaded(
                    required_datasets, 
                    st.session_state.datasets, 
                    st.session_state.db_manager if st.session_state.using_sqlite else None
                )
            
            # Now check if any are still missing
            missing_datasets = [ds for ds in required_datasets if ds not in st.session_state.datasets]
            
            if missing_datasets:
                st.error(f"Cannot calculate cross-theme overlap matrix. Missing required datasets: {', '.join(missing_datasets)}")
                st.info("Please reload data from the sidebar and select 'Download from Rebrickable' to get all necessary datasets.")
                logger.error(f"Missing datasets when calculating cross-theme overlap matrix: {missing_datasets}")
            else:
                # Calculate overlap matrix between themes
                with st.spinner("Calculating cross-theme overlap matrix..."):
                    try:
                        cross_overlap_matrix = overlap_analyzer.calculate_theme_overlap_matrix(
                            [selected_theme1, selected_theme2], 
                            st.session_state.datasets,
                            limit=cross_max_sets
                        )
                    except Exception as e:
                        error_details = traceback.format_exc()
                        st.error(f"Error calculating cross-theme overlap matrix: {str(e)}")
                        logger.error(f"Error calculating cross-theme overlap matrix: {str(e)}\n{error_details}")
                        cross_overlap_matrix = pd.DataFrame()
                
                if cross_overlap_matrix.empty:
                    st.warning("Could not calculate cross-theme overlap matrix")
                else:
                    # Get set information for better display
                    theme1_sets = data_processor.get_theme_sets(selected_theme1, st.session_state.datasets)
                    theme2_sets = data_processor.get_theme_sets(selected_theme2, st.session_state.datasets)
                    
                    set_info = pd.concat([theme1_sets, theme2_sets])
                    
                    # Create set labels
                    set_labels = {
                        row['set_num']: f"{row['set_num']}: {row['name']}" 
                        for _, row in set_info.iterrows()
                    }
                    
                    # Rename indices and columns
                    cross_overlap_matrix.index = [set_labels.get(idx, idx) for idx in cross_overlap_matrix.index]
                    cross_overlap_matrix.columns = [set_labels.get(col, col) for col in cross_overlap_matrix.columns]
                    
                    # Display heatmap
                    theme1_name = theme_map.get(selected_theme1, f"Theme {selected_theme1}")
                    theme2_name = theme_map.get(selected_theme2, f"Theme {selected_theme2}")
                    
                    st.subheader(f"Cross-Theme Overlap: {theme1_name} vs {theme2_name}")
                    st.markdown("Color represents the percentage of parts that are shared between sets (darker = more overlap)")
                    
                    fig = create_heatmap(
                        cross_overlap_matrix, 
                        f"Part Overlap Between Sets in {theme1_name} and {theme2_name}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display table for detailed view
                    with st.expander("View Cross-Theme Overlap Matrix Table"):
                        st.dataframe(cross_overlap_matrix)
                        
                        # Export option
                        st.markdown(export_to_csv(
                            cross_overlap_matrix.reset_index(), 
                            f"cross_overlap_{selected_theme1}_{selected_theme2}.csv"
                        ), unsafe_allow_html=True)
        elif not st.session_state.loaded_data and tab_selection == "Part Overlap Analysis":
            st.info("Please load data from the sidebar first to enable part overlap analysis.")

    # Tab 3: Set Purchase Optimizer
    with tab3:
        st.header("Set Purchase Optimizer")
        # Update tab selection
        tab_selection = "Set Purchase Optimizer"
        
        st.markdown("""
        **What is this?** This tool recommends the most efficient combination of LEGO sets to purchase to maximize 
        the coverage of parts from a specific theme while minimizing costs.
        
        **Why is this useful?** Instead of buying many sets with redundant parts, this tool helps you:
        - Get the maximum variety of parts for your money
        - Identify which sets give you the best "bang for your buck"
        - Plan purchases strategically to build more models with fewer sets
        
        **How to use:** Select a theme you're interested in, adjust the maximum number of sets to consider, 
        and click "Find Optimal Set Combination" to get recommendations.
        """)
    
        if st.session_state.loaded_data:
            theme_map = data_processor.get_theme_name_mapping(st.session_state.datasets)
            theme_options = [(theme_id, name) for theme_id, name in theme_map.items()]
            theme_options.sort(key=lambda x: x[1])  # Sort by theme name
            
            st.subheader("Find Optimal Sets to Purchase")
            
            col1, col2 = st.columns(2)
            
            selected_theme_id = col1.selectbox(
                "Select Theme",
                options=[t[0] for t in theme_options],
                format_func=lambda x: next((t[1] for t in theme_options if t[0] == x), "Unknown"),
                key="optimizer_theme"
            )
            
            max_sets = col2.slider(
                "Maximum Sets to Recommend",
                min_value=1,
                max_value=10,
                value=3,
                step=1,
                key="optimizer_max_sets"
            )
            
            if st.button("Find Optimal Sets"):
                # First ensure we have all required datasets
                required_datasets = ['sets', 'inventories', 'inventory_parts', 'parts']
                
                # Let the data processor try to load any missing datasets
                with st.spinner("Checking required datasets for set optimization..."):
                    st.session_state.datasets = data_processor.ensure_datasets_loaded(
                        required_datasets, 
                        st.session_state.datasets, 
                        st.session_state.db_manager if st.session_state.using_sqlite else None
                    )
                
                # Now check if any are still missing
                missing_datasets = [ds for ds in required_datasets if ds not in st.session_state.datasets]
                
                if missing_datasets:
                    st.error(f"Cannot find optimal sets. Missing required datasets: {', '.join(missing_datasets)}")
                    st.info("Please reload data from the sidebar and select 'Download from Rebrickable' to get all necessary datasets.")
                    logger.error(f"Missing datasets when finding optimal sets: {missing_datasets}")
                else:
                    with st.spinner("Finding optimal sets..."):
                        optimization_result = set_optimizer.find_minimal_sets_for_theme(
                            selected_theme_id,
                            st.session_state.datasets,
                            max_sets=max_sets
                        )
                    
                    if not optimization_result['selected_sets']:
                        st.warning(f"No sets found for theme {theme_map.get(selected_theme_id, selected_theme_id)}")
                    else:
                        # Display results
                        theme_name = theme_map.get(selected_theme_id, f"Theme {selected_theme_id}")
                        st.subheader(f"Optimal Sets for {theme_name}")
                        
                        # Summary metrics
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Parts in Theme", optimization_result['total_parts'])
                        col2.metric("Unique Parts Covered", optimization_result['unique_parts'])
                        col3.metric("Coverage Percentage", f"{optimization_result['coverage']:.1f}%")
                        
                        # Create dataframe from results
                        selected_sets_df = pd.DataFrame(optimization_result['selected_sets'])
                        
                        if not selected_sets_df.empty:
                            # Display selected sets
                            st.subheader("Recommended Sets")
                            st.markdown("These sets provide the maximum coverage of parts for the selected theme:")
                            
                            # Add coverage percentage column
                            selected_sets_df['coverage_percentage'] = selected_sets_df['new_parts'] / optimization_result['total_parts'] * 100
                            
                            # Display table
                            st.dataframe(
                                selected_sets_df[['set_num', 'name', 'year', 'new_parts', 'total_parts', 'coverage_percentage']]
                                .rename(columns={
                                    'new_parts': 'Unique Parts Added',
                                    'total_parts': 'Total Parts in Set',
                                    'coverage_percentage': 'Coverage %',
                                    'set_num': 'Set Number',
                                    'name': 'Set Name',
                                    'year': 'Year'
                                })
                                .style.format({
                                    'Coverage %': '{:.1f}%'
                                })
                            )
                        
                        # Create stacked bar chart for coverage
                        cumulative_coverage = []
                        current_coverage = 0
                        
                        for _, row in selected_sets_df.iterrows():
                            current_coverage += row['new_parts']
                            cumulative_coverage.append({
                                'Set': row['name'],
                                'Coverage': current_coverage / optimization_result['total_parts'] * 100
                            })
                        
                        coverage_df = pd.DataFrame(cumulative_coverage)
                        
                        fig = px.line(
                            coverage_df,
                            x='Set',
                            y='Coverage',
                            title=f"Cumulative Part Coverage for {theme_name}",
                            markers=True
                        )
                        
                        fig.update_layout(
                            yaxis_title="Coverage Percentage (%)",
                            xaxis_title="Sets (in order of addition)",
                            yaxis=dict(range=[0, 100])
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Export option
                        st.markdown(export_to_csv(
                            selected_sets_df, 
                            f"optimal_sets_{selected_theme_id}.csv"
                        ), unsafe_allow_html=True)
                
            st.subheader("Check What You Can Build")
            st.markdown("""
            Using your current inventory, see which sets in a theme you can build completely, 
            partially, or what parts you're missing.
            """)
            
            col1, col2 = st.columns(2)
            
            buildable_theme_id = col1.selectbox(
                "Select Theme",
                options=[t[0] for t in theme_options],
                format_func=lambda x: next((t[1] for t in theme_options if t[0] == x), "Unknown"),
                key="buildable_theme"
            )
            
            if col2.button("Analyze Buildability"):
                # First ensure we have all required datasets
                required_datasets = ['sets', 'inventories', 'inventory_parts', 'parts']
                
                # Let the data processor try to load any missing datasets
                with st.spinner("Checking required datasets for buildability analysis..."):
                    st.session_state.datasets = data_processor.ensure_datasets_loaded(
                        required_datasets, 
                        st.session_state.datasets, 
                        st.session_state.db_manager if st.session_state.using_sqlite else None
                    )
                
                # Now check if any are still missing
                missing_datasets = [ds for ds in required_datasets if ds not in st.session_state.datasets]
                
                if missing_datasets:
                    st.error(f"Cannot analyze buildability. Missing required datasets: {', '.join(missing_datasets)}")
                    st.info("Please reload data from the sidebar and select 'Download from Rebrickable' to get all necessary datasets.")
                    logger.error(f"Missing datasets when analyzing buildability: {missing_datasets}")
                elif not st.session_state.user_parts:
                    st.error("No parts in your inventory. Please add parts in the Theme Predictor tab first.")
                    logger.warning("Attempted to analyze buildability with empty user parts inventory")
                else:
                    with st.spinner("Analyzing buildability..."):
                        buildability_results = set_optimizer.analyze_buildability(
                            st.session_state.user_parts,
                            buildable_theme_id,
                            st.session_state.datasets
                        )
                    
                    if not buildability_results['buildable_sets'] and not buildability_results['missing_parts']:
                        st.warning(f"No sets found for theme {theme_map.get(buildable_theme_id, buildable_theme_id)}")
                    else:
                        theme_name = theme_map.get(buildable_theme_id, f"Theme {buildable_theme_id}")
                        
                        # Summary statistics
                        total_sets = len(buildability_results['buildable_sets']) + len(buildability_results['missing_parts'])
                        buildable_pct = len(buildability_results['buildable_sets']) / total_sets * 100 if total_sets > 0 else 0
                        
                        st.subheader(f"Buildability Analysis for {theme_name}")
                        
                        # Summary metrics
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Sets in Theme", total_sets)
                        col2.metric("Fully Buildable Sets", len(buildability_results['buildable_sets']))
                        col3.metric("Buildable Percentage", f"{buildable_pct:.1f}%")
                        
                        # Create pie chart
                        buildability_data = {
                            'Status': ['Buildable', 'Missing Parts'],
                            'Count': [len(buildability_results['buildable_sets']), len(buildability_results['missing_parts'])]
                        }
                        
                        if buildability_data['Count'][0] > 0 or buildability_data['Count'][1] > 0:
                            fig = create_donut_chart(
                                buildability_data['Count'],
                                buildability_data['Status'],
                                f"Buildability Status for {theme_name}"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Show buildable sets
                        if buildability_results['buildable_sets']:
                            st.subheader("Fully Buildable Sets")
                            st.markdown("These sets can be built completely with your current inventory:")
                            
                            buildable_df = pd.DataFrame(buildability_results['buildable_sets'])
                            buildable_df = buildable_df.sort_values('year', ascending=False)
                            
                            st.dataframe(
                                buildable_df[['set_num', 'name', 'year', 'num_parts']]
                                .rename(columns={
                                    'set_num': 'Set Number',
                                    'name': 'Set Name',
                                    'year': 'Year',
                                    'num_parts': 'Total Parts'
                                })
                            )
                        
                        # Show sets with missing parts
                        if buildability_results['missing_parts']:
                            st.subheader("Sets with Missing Parts")
                            st.markdown("These sets require additional parts to be buildable:")
                            
                            # Create a summary of missing parts sets
                            missing_sets_summary = []
                            
                            for set_num, info in buildability_results['missing_parts'].items():
                                missing_sets_summary.append({
                                    'set_num': set_num,
                                    'name': info['set_name'],
                                    'buildability': info['buildability'],
                                    'missing_part_count': len(info['missing_parts']),
                                    'year': info['year'],
                                    'total_parts': info['num_parts']
                                })
                            
                            missing_df = pd.DataFrame(missing_sets_summary)
                            missing_df = missing_df.sort_values('buildability', ascending=False)
                            
                            st.dataframe(
                                missing_df[['set_num', 'name', 'buildability', 'missing_part_count', 'year', 'total_parts']]
                                .rename(columns={
                                    'set_num': 'Set Number',
                                    'name': 'Set Name',
                                    'buildability': 'Buildability %',
                                    'missing_part_count': 'Missing Parts Count',
                                    'year': 'Year',
                                    'total_parts': 'Total Parts'
                                })
                                .style.format({
                                    'Buildability %': '{:.1f}%'
                                })
                            )
                            
                            # Allow user to select a set to see missing parts
                            if not missing_df.empty:
                                selected_set = st.selectbox(
                                    "Select a set to see missing parts",
                                    options=missing_df['set_num'].tolist(),
                                    format_func=lambda x: f"{x}: {next((row['name'] for _, row in missing_df.iterrows() if row['set_num'] == x), 'Unknown')}"
                                )
                                
                                if selected_set in buildability_results['missing_parts']:
                                    missing_parts = buildability_results['missing_parts'][selected_set]['missing_parts']
                                    
                                    st.subheader(f"Missing Parts for {selected_set}")
                                    
                                    missing_parts_df = pd.DataFrame(missing_parts)
                                    missing_parts_df = missing_parts_df.sort_values('shortfall', ascending=False)
                                    
                                    st.dataframe(
                                        missing_parts_df[['part_num', 'name', 'required', 'available', 'shortfall']]
                                        .rename(columns={
                                            'part_num': 'Part Number',
                                            'name': 'Part Name',
                                            'required': 'Required',
                                            'available': 'Available',
                                            'shortfall': 'Shortfall'
                                        })
                                    )
                                    
                                    # Export option
                                    st.markdown(export_to_csv(
                                        missing_parts_df, 
                                        f"missing_parts_{selected_set}.csv"
                                    ), unsafe_allow_html=True)
        elif not st.session_state.loaded_data and tab_selection == "Set Purchase Optimizer":
            st.info("Please load data from the sidebar first to enable set purchase optimization.")

# Tab 4: Theme Requirements
    with tab4:
        st.header("Theme Requirements Analysis")
        # Update tab selection
        tab_selection = "Theme Requirements Analysis"
        st.markdown("""
        **What is this?** This tab analyzes what parts are most characteristic and unique to specific LEGO themes.
        
        **Why is this useful?** This analysis helps you:
        - Understand which parts define a specific theme's appearance or function
        - Identify parts to acquire if you want to build in a particular theme's style
        - Find the "signature parts" that make a theme distinct from others
        - See how well your current inventory matches different themes
        
        **How to use:** Select a theme and click "Analyze Theme Requirements" to see the most distinctive parts 
        for that theme, their frequencies, and how unique they are to that theme.
        """)
    
        if st.session_state.loaded_data:
            theme_map = data_processor.get_theme_name_mapping(st.session_state.datasets)
            theme_options = [(theme_id, name) for theme_id, name in theme_map.items()]
            theme_options.sort(key=lambda x: x[1])  # Sort by theme name
            
            # Add new section for inventory theme analysis
            st.subheader("Analyze Your Inventory's Theme Match")
            
            if st.button("Analyze Themes for My Inventory"):
                if len(st.session_state.user_parts) > 0:
                    with st.spinner("Analyzing themes that match your inventory..."):
                        try:
                            # Use theme predictor to suggest themes
                            prediction = st.session_state.theme_predictor.predict(st.session_state.user_parts)
                            if prediction:
                                st.subheader("Potential Themes to Build")
                                st.write("Based on your parts inventory, you might be able to build sets from these themes:")
                                
                                # Display top themes with confidence
                                theme_df = pd.DataFrame(prediction['top_themes'])
                                theme_df['confidence'] = theme_df['confidence'] * 100
                                
                                # Create a bar chart for visual representation
                                fig = px.bar(
                                    theme_df,
                                    x='theme_name',
                                    y='confidence',
                                    title="Themes that Match Your Inventory",
                                    color='confidence',
                                    color_continuous_scale='greens',
                                    text='confidence'
                                )
                                
                                fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Display as a table
                                st.dataframe(
                                    theme_df.rename(columns={
                                        'theme_name': 'Theme',
                                        'confidence': 'Match Confidence (%)'
                                    }).style.format({
                                        'Match Confidence (%)': '{:.1f}%'
                                    })
                                )
                                
                                # Recommend next steps
                                top_theme_id = prediction['theme_id']
                                top_theme_name = prediction['theme_name']
                                
                                st.subheader("Next Steps")
                                st.markdown(f"""
                                To get started building with your inventory:
                                
                                1. Try analyzing the buildability of **{top_theme_name}** in the "Set Purchase Optimizer" tab
                                2. Check which parts you're missing for specific sets
                                3. Consider adding more parts to your inventory that complement your current collection
                                """)
                        except Exception as e:
                            st.error(f"Could not analyze themes: {e}")
                else:
                    st.warning("Please add parts to your inventory first. Use the 'Your LEGO Inventory' section above.")
        
        st.markdown("---")
        st.subheader("Analyze Theme Part Requirements")
        
        col1, col2 = st.columns(2)
        
        requirements_theme_id = col1.selectbox(
            "Select Theme",
            options=[t[0] for t in theme_options],
            format_func=lambda x: next((t[1] for t in theme_options if t[0] == x), "Unknown"),
            key="requirements_theme"
        )
        
        include_unique_only = col2.checkbox("Show Unique Parts Only", value=False)
        
        if st.button("Analyze Theme Requirements"):
            with st.spinner("Analyzing theme requirements..."):
                # Get theme statistics
                theme_stats = overlap_analyzer.get_theme_part_counts(
                    requirements_theme_id,
                    st.session_state.datasets
                )
                
                if theme_stats['total_sets'] == 0:
                    st.warning(f"No sets found for theme {theme_map.get(requirements_theme_id, requirements_theme_id)}")
                else:
                    theme_name = theme_map.get(requirements_theme_id, f"Theme {requirements_theme_id}")
                    
                    st.subheader(f"Part Requirements for {theme_name}")
                    
                    # Summary metrics
                    col1, col2 = st.columns(2)
                    col1.metric("Total Sets in Theme", theme_stats['total_sets'])
                    col2.metric("Unique Parts in Theme", theme_stats['total_unique_parts'])
                    
                    # Get part frequency data
                    part_freq = theme_stats['part_frequency']
                    
                    if include_unique_only:
                        # Get unique parts for this theme
                        unique_parts = data_processor.get_unique_parts_for_theme(
                            requirements_theme_id,
                            st.session_state.datasets
                        )
                        
                        if unique_parts.empty:
                            st.info("No unique parts found for this theme")
                        else:
                            # Merge with part frequency data to get set percentages
                            unique_parts_df = unique_parts.merge(
                                part_freq[['part_num', 'sets_percentage']],
                                on='part_num',
                                how='left'
                            )
                            
                            st.subheader("Unique Parts for This Theme")
                            st.markdown("These parts appear in this theme but not in others:")
                            
                            display_df = unique_parts_df[['part_num', 'name', 'sets_percentage']]
                            display_df = display_df.sort_values('sets_percentage', ascending=False)
                            
                            st.dataframe(
                                display_df.rename(columns={
                                    'part_num': 'Part Number',
                                    'name': 'Part Name',
                                    'sets_percentage': 'Sets Containing Part (%)'
                                }).style.format({
                                    'Sets Containing Part (%)': '{:.1f}%'
                                })
                            )
                            
                            # Export option
                            st.markdown(export_to_csv(
                                display_df, 
                                f"unique_parts_{requirements_theme_id}.csv"
                            ), unsafe_allow_html=True)
                            
                            # Create bar chart of top unique parts
                            top_parts = display_df.head(15)
                            
                            fig = px.bar(
                                top_parts,
                                x='part_num',
                                y='sets_percentage',
                                title=f"Top Unique Parts in {theme_name}",
                                labels={
                                    'part_num': 'Part Number',
                                    'sets_percentage': 'Sets Containing Part (%)'
                                },
                                color='sets_percentage',
                                color_continuous_scale='viridis'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        # Show parts by frequency across sets
                        if part_freq.empty:
                            st.info("No part frequency data available for this theme")
                        else:
                            st.subheader("Part Frequency Across Sets")
                            st.markdown("These are the most common parts in this theme (ranked by % of sets containing them):")
                            
                            display_df = part_freq[['part_num', 'name', 'sets_percentage', 'total_quantity']]
                            display_df = display_df.sort_values('sets_percentage', ascending=False)
                            
                            st.dataframe(
                                display_df.head(50).rename(columns={
                                    'part_num': 'Part Number',
                                    'name': 'Part Name',
                                    'sets_percentage': 'Sets Containing Part (%)',
                                    'total_quantity': 'Total Quantity Across Theme'
                                }).style.format({
                                    'Sets Containing Part (%)': '{:.1f}%'
                                })
                            )
                            
                            # Export option
                            st.markdown(export_to_csv(
                                display_df, 
                                f"part_frequency_{requirements_theme_id}.csv"
                            ), unsafe_allow_html=True)
                            
                            # Create bar chart of top parts
                            top_parts = display_df.head(15)
                            
                            fig = px.bar(
                                top_parts,
                                x='part_num',
                                y='sets_percentage',
                                title=f"Most Common Parts in {theme_name}",
                                labels={
                                    'part_num': 'Part Number',
                                    'sets_percentage': 'Sets Containing Part (%)'
                                },
                                color='sets_percentage',
                                color_continuous_scale='viridis'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Show breakdown by set count
                            st.subheader("Sets Per Part Distribution")
                            
                            # Create distribution data
                            bins = [0, 10, 25, 50, 75, 90, 100]
                            labels = ['0-10%', '10-25%', '25-50%', '50-75%', '75-90%', '90-100%']
                            
                            part_freq['bin'] = pd.cut(part_freq['sets_percentage'], bins=bins, labels=labels)
                            
                            distribution = part_freq['bin'].value_counts().reset_index()
                            distribution.columns = ['Percentage of Sets', 'Number of Parts']
                            distribution = distribution.sort_values('Percentage of Sets')
                            
                            fig = px.pie(
                                distribution,
                                values='Number of Parts',
                                names='Percentage of Sets',
                                title='Distribution of Parts by Set Frequency',
                                hole=0.4
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
        elif not st.session_state.loaded_data and tab_selection == "Theme Requirements Analysis":
            st.info("Please load data from the sidebar first to enable theme requirements analysis.")

# Tab 5: Data Browser
    with tab5:
        st.header("Rebrickable Data Browser")
        # Update tab selection
        tab_selection = "Data Browser"
        st.markdown("""
        **What is this?** This tab provides direct access to the raw LEGO datasets from Rebrickable.
        
        **Why is this useful?** This browser helps you:
        - Explore the datasets directly to understand their structure
        - Look up specific set numbers, part numbers, or theme information
        - Export data for use in other applications
        - Analyze column statistics and distributions
        
        **How to use:** Select a dataset from the dropdown, optionally filter the data,
        and view the results. You can also see column statistics and export data as CSV.
        """)
        
        if st.session_state.loaded_data:
            # Database mode selector
            if 'using_sqlite' in st.session_state and st.session_state.using_sqlite:
                data_source_type = st.radio(
                    "Data Source",
                    ["SQLite Database", "Memory Datasets"],
                    index=0
                )
            else:
                data_source_type = "Memory Datasets"
            
            if data_source_type == "SQLite Database" and 'using_sqlite' in st.session_state and st.session_state.using_sqlite:
                # Get tables from SQLite
                tables_query = """
                SELECT name FROM sqlite_master 
                WHERE type='table' 
                ORDER BY name;
                """
                tables_df = db_manager.query_to_dataframe(tables_query)
                
                if not tables_df.empty:
                    # Let user select a table to view
                    available_tables = tables_df['name'].tolist()
                    
                    selected_table = st.selectbox(
                        "Select a table to view",
                        options=available_tables
                    )
                
                # Get sample of table data
                st.subheader(f"{selected_table.capitalize()} Table")
                
                # Allow filtering number of rows
                num_rows = st.slider(
                    "Number of rows to display", 
                    min_value=5, 
                    max_value=100, 
                    value=10,
                    step=5
                )
                
                # Get row count
                count_query = f"SELECT COUNT(*) as count FROM {selected_table}"
                count_df = db_manager.query_to_dataframe(count_query)
                total_rows = count_df.iloc[0]['count'] if not count_df.empty else 0
                
                # Display dataset info
                st.info(f"Total rows in table: {total_rows}")
                
                # Get sample data
                sample_query = f"SELECT * FROM {selected_table} LIMIT {num_rows}"
                df = db_manager.query_to_dataframe(sample_query)
                
                if not df.empty:
                    # Display column names
                    st.write(f"Columns: {', '.join(df.columns.tolist())}")
                    
                    # Display the data
                    st.dataframe(df)
                    
                    # Option to see column statistics
                    if st.checkbox("Show column statistics"):
                        try:
                            # Get schema
                            schema_query = f"PRAGMA table_info({selected_table})"
                            schema_df = db_manager.query_to_dataframe(schema_query)
                            
                            if not schema_df.empty:
                                st.subheader("Table Schema")
                                st.dataframe(schema_df[['name', 'type', 'pk', 'notnull']])
                            
                            # Calculate statistics for numeric columns
                            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                            
                            if numeric_cols:
                                st.subheader("Numeric Column Statistics")
                                stats_results = []
                                
                                for col in numeric_cols:
                                    stats_query = f"""
                                    SELECT 
                                        '{col}' as column_name,
                                        COUNT({col}) as count,
                                        AVG({col}) as mean,
                                        MIN({col}) as min,
                                        MAX({col}) as max
                                    FROM {selected_table}
                                    """
                                    stats_df = db_manager.query_to_dataframe(stats_query)
                                    if not stats_df.empty:
                                        stats_results.append(stats_df)
                                
                                if stats_results:
                                    combined_stats = pd.concat(stats_results)
                                    st.dataframe(combined_stats)
                            
                            # Show counts for categorical columns
                            categorical_cols = [col for col in df.columns if col not in numeric_cols][:5]  # Limit to first 5
                            
                            if categorical_cols:
                                st.subheader("Top Values in Categorical Columns")
                                
                                for col in categorical_cols:
                                    st.write(f"Top 5 values in '{col}':")
                                    top_values_query = f"""
                                    SELECT {col}, COUNT(*) as count
                                    FROM {selected_table}
                                    WHERE {col} IS NOT NULL
                                    GROUP BY {col}
                                    ORDER BY count DESC
                                    LIMIT 5
                                    """
                                    top_values_df = db_manager.query_to_dataframe(top_values_query)
                                    if not top_values_df.empty:
                                        st.dataframe(top_values_df)
                        except Exception as e:
                            st.error(f"Could not generate statistics: {str(e)}")
                    
                    # Show export option
                    if st.button(f"Export Full {selected_table} Table to CSV"):
                        try:
                            full_query = f"SELECT * FROM {selected_table}"
                            full_df = db_manager.query_to_dataframe(full_query)
                            
                            if not full_df.empty:
                                st.markdown(export_to_csv(full_df, f"{selected_table}.csv"), unsafe_allow_html=True)
                                st.success(f"Data prepared for download ({len(full_df)} rows)")
                            else:
                                st.warning("No data to export")
                        except Exception as e:
                            st.error(f"Error exporting data: {str(e)}")
                    
                    # Show database info
                    with st.expander("Database Information"):
                        st.markdown("""
                        This data is stored in a SQLite database.
                        
                        **Database location:**
                        """)
                        st.code(f"{db_manager.db_path}")
                        
                        # Check if file exists
                        if os.path.exists(db_manager.db_path):
                            file_stats = os.stat(db_manager.db_path)
                            st.write(f"Database size: {file_stats.st_size / (1024*1024):.2f} MB")
                            st.write(f"Last modified: {datetime.fromtimestamp(file_stats.st_mtime)}")
                else:
                    st.warning(f"Table '{selected_table}' appears to be empty")
            else:
                st.warning("No tables found in the database")
                
        else:  # Use in-memory datasets
            # Get available datasets
            available_datasets = list(st.session_state.datasets.keys())
            available_datasets.sort()
            
            # Let user select a dataset to view
            selected_dataset = st.selectbox(
                "Select a dataset to view",
                options=available_datasets
            )
            
            if selected_dataset in st.session_state.datasets:
                df = st.session_state.datasets[selected_dataset]
                
                st.subheader(f"{selected_dataset.capitalize()} Dataset")
                
                # Display dataset info
                st.info(f"Total rows: {len(df)}, Columns: {', '.join(df.columns.tolist())}")
                
                # Allow filtering number of rows
                num_rows = st.slider(
                    "Number of rows to display", 
                    min_value=5, 
                    max_value=min(100, len(df)), 
                    value=10,
                    step=5
                )
                
                # Display the data
                st.dataframe(df.head(num_rows))
                
                # Quick Visualization section removed as requested
                
                # Option to see column statistics
                if st.checkbox("Show column statistics"):
                    try:
                        # Calculate statistics for numeric columns
                        st.write("Numeric Column Statistics:")
                        numeric_stats = df.describe().transpose()
                        st.dataframe(numeric_stats)
                        
                        # Show counts for categorical columns
                        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                        if categorical_cols:
                            st.write("Top values in categorical columns:")
                            for col in categorical_cols[:5]:  # Limit to first 5 categorical columns
                                st.write(f"Top 5 values in '{col}':")
                                st.write(df[col].value_counts().head(5))
                    except Exception as e:
                        st.error(f"Could not generate statistics: {str(e)}")
                
                # Export option
                st.markdown(export_to_csv(df, f"{selected_dataset}.csv"), unsafe_allow_html=True)
                
                # Show data sources info
                with st.expander("Data Source Information"):
                    st.markdown("""
                    This data is sourced from Rebrickable.com, which provides detailed LEGO set 
                    and part information. 
                    
                    **File locations:**
                    """)
                    st.code(f"./lego_data/{selected_dataset}.csv")
                    
                    # Check if file exists
                    if os.path.exists(f"./lego_data/{selected_dataset}.csv"):
                        file_stats = os.stat(f"./lego_data/{selected_dataset}.csv")
                        st.write(f"File size: {file_stats.st_size / 1024:.1f} KB")
                        st.write(f"Last modified: {datetime.fromtimestamp(file_stats.st_mtime)}")
            elif tab_selection == "Data Browser":
                st.info("Please load data from the sidebar first to browse Rebrickable datasets.")

# Tab 6: SQL Query 
    with tab6:
        st.header("SQL Query Tool")
        # Update tab selection
        tab_selection = "SQL Query"
        st.markdown("""
        **What is this?** This tab provides a direct SQL interface to the LEGO database for advanced analysis.
        
        **Why is this useful?** The SQL Query tool allows you to:
        - Perform complex custom analyses not available in other tabs
        - Cross-reference multiple tables simultaneously
        - Create specific data views tailored to your research interests
        - Gain deeper insights through flexible database queries
        
        **How to use:** Select a sample query from the dropdown or write your own SQL query,
        then click "Run Query" to execute it. Results can be viewed and exported as CSV.
        """)
        
        if st.session_state.loaded_data and ('using_sqlite' in st.session_state and st.session_state.using_sqlite):
            # Provide a schema reference
            with st.expander("Database Schema Reference"):
                st.markdown("""
            ### Database Tables and Schemas
            
            **themes**
            - id: INTEGER PRIMARY KEY
            - name: TEXT
            - parent_id: INTEGER (references themes.id)
            
            **sets**
            - set_num: TEXT PRIMARY KEY
            - name: TEXT
            - year: INTEGER
            - theme_id: INTEGER (references themes.id)
            - num_parts: INTEGER
            
            **parts**
            - part_num: TEXT PRIMARY KEY
            - name: TEXT
            - part_cat_id: INTEGER (references part_categories.id)
            
            **colors**
            - id: INTEGER PRIMARY KEY
            - name: TEXT
            - rgb: TEXT
            - is_trans: TEXT
            
            **part_categories**
            - id: INTEGER PRIMARY KEY
            - name: TEXT
            
            **inventories**
            - id: INTEGER PRIMARY KEY
            - version: INTEGER
            - set_num: TEXT (references sets.set_num)
            
            **inventory_parts**
            - inventory_id: INTEGER (references inventories.id)
            - part_num: TEXT (references parts.part_num)
            - color_id: INTEGER (references colors.id)
            - quantity: INTEGER
            - is_spare: TEXT
            """)
            
                # Display sample queries
                st.markdown("""
                ### Sample Queries
                
                **Find top 10 sets with the most parts:**
                ```sql
                SELECT set_num, name, year, theme_id, num_parts
                FROM sets
                ORDER BY num_parts DESC
                LIMIT 10;
                ```
                
                **Find parts used in multiple themes:**
                ```sql
                SELECT p.part_num, p.name, COUNT(DISTINCT s.theme_id) as theme_count
                FROM inventory_parts ip
                JOIN inventories i ON ip.inventory_id = i.id
                JOIN sets s ON i.set_num = s.set_num
                JOIN parts p ON ip.part_num = p.part_num
                GROUP BY p.part_num, p.name
                HAVING theme_count > 5
                ORDER BY theme_count DESC
                LIMIT 20;
                ```
                
                **Find sets by year and theme:**
                ```sql
                SELECT s.set_num, s.name, s.year, s.num_parts, t.name as theme_name
                FROM sets s
                JOIN themes t ON s.theme_id = t.id
                WHERE s.year = 2020 AND t.name LIKE '%Star Wars%'
                ORDER BY s.num_parts DESC;
                ```
                """)
        
            # Sample queries dropdown
            sample_queries = {
                "Custom Query": "",
                "Top 10 largest sets": """
                    SELECT set_num, name, year, theme_id, num_parts
                    FROM sets
                    ORDER BY num_parts DESC
                    LIMIT 10;
                """,
                "Parts used across many themes": """
                    SELECT p.part_num, p.name, COUNT(DISTINCT s.theme_id) as theme_count
                    FROM inventory_parts ip
                    JOIN inventories i ON ip.inventory_id = i.id
                    JOIN sets s ON i.set_num = s.set_num
                    JOIN parts p ON ip.part_num = p.part_num
                    GROUP BY p.part_num, p.name
                    HAVING theme_count > 5
                    ORDER BY theme_count DESC
                    LIMIT 20;
                """,
                "Most common parts": """
                    SELECT p.part_num, p.name, COUNT(*) as set_count, SUM(ip.quantity) as total_quantity
                    FROM inventory_parts ip
                    JOIN parts p ON ip.part_num = p.part_num
                    GROUP BY p.part_num, p.name
                    ORDER BY set_count DESC
                    LIMIT 20;
                """,
                "Theme distribution by year": """
                    SELECT t.name as theme_name, s.year, COUNT(*) as set_count, SUM(s.num_parts) as total_parts
                    FROM sets s
                    JOIN themes t ON s.theme_id = t.id
                    GROUP BY t.name, s.year
                    ORDER BY s.year DESC, set_count DESC;
                """,
                "Sets with specific part": """
                    SELECT s.set_num, s.name, s.year, t.name as theme_name, ip.quantity
                    FROM inventory_parts ip
                    JOIN inventories i ON ip.inventory_id = i.id
                    JOIN sets s ON i.set_num = s.set_num
                    JOIN themes t ON s.theme_id = t.id
                    WHERE ip.part_num = '3001'  -- Brick 2 x 4
                    ORDER BY ip.quantity DESC
                    LIMIT 20;
                """
        }
        
            # Select query template
            selected_query_name = st.selectbox(
                "Select a sample query or write your own",
                options=list(sample_queries.keys())
            )
            
            # Display query editor
            default_query = sample_queries[selected_query_name].strip()
            query = st.text_area("SQL Query", value=default_query, height=200)
            
            # Run query button
            if st.button("Run Query"):
                if query.strip():
                    with st.spinner("Executing query..."):
                        try:
                            # Execute the query
                            result_df = db_manager.query_to_dataframe(query)
                            
                            if result_df is not None and not result_df.empty:
                                st.success(f"Query executed successfully. {len(result_df)} rows returned.")
                                
                                # Display results
                                st.subheader("Query Results")
                                st.dataframe(result_df)
                                
                                # Export option
                                st.markdown(export_to_csv(result_df, "query_results.csv"), unsafe_allow_html=True)
                                
                                # Quick Visualization section removed as requested
                            else:
                                st.info("Query executed successfully, but no results were returned.")
                        except Exception as e:
                            st.error(f"Error executing query: {str(e)}")
                else:
                    st.warning("Please enter a valid SQL query.")
        else:
            if not st.session_state.loaded_data and tab_selection == "SQL Query":
                st.info("Please load data from the sidebar first to enable SQL queries.")
            else:
                st.warning("SQL queries require the SQLite database. Please load data with the 'Download from Rebrickable' option to create the database.")

# We already have an "About this Application" section in the landing page,
# so we don't need to show it again here to avoid duplication

# Show data loading status
if st.session_state.loaded_data:
    st.success("Data loaded successfully!")
    
    # Show summary of loaded datasets
    with st.expander("Data Source Summary"):
        dataset_info = []
        for name, df in st.session_state.datasets.items():
            if not df.empty:
                dataset_info.append({
                    'Dataset': name,
                    'Rows': len(df),
                    'Columns': len(df.columns)
                })
        
        if dataset_info:
            st.table(pd.DataFrame(dataset_info))
        else:
            st.info("No datasets loaded")
else:
    # Display a more distinct call-to-action when data isn't loaded
    st.markdown("""
    <div class="banner">
        <h3>Ready to Explore? 🚀</h3>
        <p>Click the <b>Load Data</b> button in the sidebar to activate all features and begin your LEGO analysis journey!</p>
    </div>
    """, unsafe_allow_html=True)



# Footer HTML - added outside the conditional blocks so it appears on all pages
st.markdown("""
<div class="footer">
    <p>© Copyright 2025 - Fairfield University Dolan School of Business</p>
    <p>Leading of Analytics</p>
    <p>Made with <span class="heart">❤</span> by Aslam, Aishwarya, Sidharth</p>
</div>
""", unsafe_allow_html=True)
