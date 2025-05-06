import os
import sqlite3
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

class DatabaseManager:
    """
    Manage SQLite database operations for the LEGO Analyzer application
    """
    def __init__(self, db_path="./lego_data/lego.db"):
        """
        Initialize the database manager
        
        Args:
            db_path (str): Path to the SQLite database file
        """
        self.db_path = db_path
        self.ensure_db_directory()
        
    def ensure_db_directory(self):
        """Ensure the directory for the database exists"""
        db_dir = os.path.dirname(self.db_path)
        os.makedirs(db_dir, exist_ok=True)
    
    def get_connection(self):
        """Get a connection to the database"""
        return sqlite3.connect(self.db_path)
    
    def db_exists(self):
        """Check if the database file exists"""
        return os.path.exists(self.db_path)
    
    def is_data_fresh(self, max_age_days=7):
        """
        Check if the database was updated recently
        
        Args:
            max_age_days (int): Maximum age in days for data to be considered fresh
            
        Returns:
            bool: Whether data is fresh
        """
        if not self.db_exists():
            return False
            
        # Check modification time of database file
        mod_time = datetime.fromtimestamp(os.path.getmtime(self.db_path))
        age = datetime.now() - mod_time
        
        return age.days < max_age_days
        
    def create_tables(self):
        """Create all required tables if they don't exist"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Define all tables with their schemas
        tables = {
            "themes": """
                CREATE TABLE IF NOT EXISTS themes (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    parent_id INTEGER
                )
            """,
            "sets": """
                CREATE TABLE IF NOT EXISTS sets (
                    set_num TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    year INTEGER,
                    theme_id INTEGER,
                    num_parts INTEGER,
                    FOREIGN KEY (theme_id) REFERENCES themes (id)
                )
            """,
            "colors": """
                CREATE TABLE IF NOT EXISTS colors (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    rgb TEXT,
                    is_trans TEXT
                )
            """,
            "part_categories": """
                CREATE TABLE IF NOT EXISTS part_categories (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL
                )
            """,
            "parts": """
                CREATE TABLE IF NOT EXISTS parts (
                    part_num TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    part_cat_id INTEGER,
                    FOREIGN KEY (part_cat_id) REFERENCES part_categories (id)
                )
            """,
            "inventories": """
                CREATE TABLE IF NOT EXISTS inventories (
                    id INTEGER PRIMARY KEY,
                    version INTEGER,
                    set_num TEXT,
                    FOREIGN KEY (set_num) REFERENCES sets (set_num)
                )
            """,
            "inventory_parts": """
                CREATE TABLE IF NOT EXISTS inventory_parts (
                    inventory_id INTEGER,
                    part_num TEXT,
                    color_id INTEGER,
                    quantity INTEGER NOT NULL,
                    is_spare TEXT,
                    PRIMARY KEY (inventory_id, part_num, color_id),
                    FOREIGN KEY (inventory_id) REFERENCES inventories (id),
                    FOREIGN KEY (part_num) REFERENCES parts (part_num),
                    FOREIGN KEY (color_id) REFERENCES colors (id)
                )
            """
        }
        
        # Create each table
        for table_name, create_stmt in tables.items():
            try:
                cursor.execute(create_stmt)
            except sqlite3.Error as e:
                st.error(f"Error creating {table_name} table: {str(e)}")
        
        conn.commit()
        conn.close()
    
    def insert_dataframe(self, table_name, df, if_exists='replace'):
        """
        Insert a pandas DataFrame into a database table
        
        Args:
            table_name (str): Name of the table
            df (DataFrame): DataFrame to insert
            if_exists (str): How to behave if table exists ('replace' or 'append')
        """
        if df.empty:
            st.warning(f"Skipping empty DataFrame for table {table_name}")
            return
            
        try:
            conn = self.get_connection()
            df.to_sql(table_name, conn, if_exists=if_exists, index=False)
            conn.commit()
            conn.close()
            st.success(f"Successfully loaded {len(df)} rows into {table_name} table")
        except Exception as e:
            st.error(f"Error loading data into {table_name}: {str(e)}")
    
    def query_to_dataframe(self, query, params=None):
        """
        Execute SQL query and return results as a DataFrame
        
        Args:
            query (str): SQL query to execute
            params (tuple): Parameters for the query
            
        Returns:
            DataFrame: Query results
        """
        try:
            conn = self.get_connection()
            if params:
                df = pd.read_sql_query(query, conn, params=params)
            else:
                df = pd.read_sql_query(query, conn)
            conn.close()
            return df
        except Exception as e:
            st.error(f"Error executing query: {str(e)}")
            return pd.DataFrame()
    
    def load_rebrickable_data(self, datasets):
        """
        Load Rebrickable datasets into the database
        
        Args:
            datasets (dict): Dictionary of pandas DataFrames
        """
        # Create tables if they don't exist
        self.create_tables()
        
        # Map dataset names to table names
        table_mapping = {
            "themes": "themes",
            "sets": "sets",
            "colors": "colors",
            "part_categories": "part_categories", 
            "parts": "parts",
            "inventories": "inventories",
            "inventory_parts": "inventory_parts"
        }
        
        # Insert each dataset into its corresponding table
        for dataset_name, table_name in table_mapping.items():
            if dataset_name in datasets and not datasets[dataset_name].empty:
                self.insert_dataframe(table_name, datasets[dataset_name])
    
    def get_table_data(self, table_name, limit=100):
        """
        Get data from a specific table
        
        Args:
            table_name (str): Name of the table
            limit (int): Maximum number of rows to return
            
        Returns:
            DataFrame: Table data
        """
        query = f"SELECT * FROM {table_name} LIMIT {limit}"
        return self.query_to_dataframe(query)
    
    def get_theme_sets(self, theme_id):
        """
        Get all sets for a specific theme
        
        Args:
            theme_id (int): Theme ID
            
        Returns:
            DataFrame: Sets in this theme
        """
        query = """
        SELECT * FROM sets
        WHERE theme_id = ?
        ORDER BY year DESC, name
        """
        return self.query_to_dataframe(query, (theme_id,))
    
    def get_set_parts(self, set_num):
        """
        Get all parts for a specific set
        
        Args:
            set_num (str): Set number
            
        Returns:
            DataFrame: Parts in this set with quantities
        """
        query = """
        SELECT p.part_num, p.name as part_name, ip.color_id, c.name as color_name, 
               ip.quantity, ip.is_spare
        FROM inventory_parts ip
        JOIN inventories i ON ip.inventory_id = i.id
        JOIN parts p ON ip.part_num = p.part_num
        JOIN colors c ON ip.color_id = c.id
        WHERE i.set_num = ?
        ORDER BY ip.quantity DESC
        """
        return self.query_to_dataframe(query, (set_num,))
    
    def get_parts_by_category(self, category_id=None):
        """
        Get parts filtered by category
        
        Args:
            category_id (int): Category ID or None for all
            
        Returns:
            DataFrame: Parts with their categories
        """
        if category_id:
            query = """
            SELECT p.part_num, p.name, pc.name as category
            FROM parts p
            JOIN part_categories pc ON p.part_cat_id = pc.id
            WHERE p.part_cat_id = ?
            ORDER BY p.name
            """
            return self.query_to_dataframe(query, (category_id,))
        else:
            query = """
            SELECT p.part_num, p.name, pc.name as category
            FROM parts p
            JOIN part_categories pc ON p.part_cat_id = pc.id
            ORDER BY pc.name, p.name
            LIMIT 1000
            """
            return self.query_to_dataframe(query)
    
    def get_theme_part_frequency(self, theme_id):
        """
        Get frequency of parts in a specific theme
        
        Args:
            theme_id (int): Theme ID
            
        Returns:
            DataFrame: Part frequency data
        """
        query = """
        WITH theme_sets AS (
            SELECT set_num
            FROM sets
            WHERE theme_id = ?
        ),
        theme_inventories AS (
            SELECT id
            FROM inventories
            WHERE set_num IN (SELECT set_num FROM theme_sets)
        ),
        set_count AS (
            SELECT COUNT(DISTINCT set_num) as total_sets
            FROM sets
            WHERE theme_id = ?
        )
        SELECT 
            p.part_num,
            p.name,
            COUNT(DISTINCT ip.inventory_id) as sets_containing,
            (COUNT(DISTINCT ip.inventory_id) * 100.0 / (SELECT total_sets FROM set_count)) as sets_percentage,
            SUM(ip.quantity) as total_quantity
        FROM inventory_parts ip
        JOIN parts p ON ip.part_num = p.part_num
        WHERE ip.inventory_id IN (SELECT id FROM theme_inventories)
            AND ip.is_spare = 'f'
        GROUP BY p.part_num, p.name
        ORDER BY sets_percentage DESC, total_quantity DESC
        """
        return self.query_to_dataframe(query, (theme_id, theme_id))
    
    def get_unique_theme_parts(self, theme_id):
        """
        Get parts that are unique to a specific theme
        
        Args:
            theme_id (int): Theme ID
            
        Returns:
            DataFrame: Unique parts for this theme
        """
        query = """
        WITH theme_sets AS (
            SELECT set_num
            FROM sets
            WHERE theme_id = ?
        ),
        theme_inventories AS (
            SELECT id
            FROM inventories
            WHERE set_num IN (SELECT set_num FROM theme_sets)
        ),
        theme_parts AS (
            SELECT DISTINCT ip.part_num
            FROM inventory_parts ip
            WHERE ip.inventory_id IN (SELECT id FROM theme_inventories)
                AND ip.is_spare = 'f'
        ),
        other_theme_parts AS (
            SELECT DISTINCT ip.part_num
            FROM inventory_parts ip
            JOIN inventories i ON ip.inventory_id = i.id
            JOIN sets s ON i.set_num = s.set_num
            WHERE s.theme_id != ?
                AND ip.is_spare = 'f'
        )
        SELECT 
            p.part_num,
            p.name,
            pc.name as category
        FROM theme_parts tp
        JOIN parts p ON tp.part_num = p.part_num
        LEFT JOIN part_categories pc ON p.part_cat_id = pc.id
        WHERE tp.part_num NOT IN (SELECT part_num FROM other_theme_parts)
        ORDER BY p.name
        """
        return self.query_to_dataframe(query, (theme_id, theme_id))
    
    def calculate_set_overlap(self, set_num1, set_num2):
        """
        Calculate part overlap between two sets
        
        Args:
            set_num1 (str): First set number
            set_num2 (str): Second set number
            
        Returns:
            dict: Overlap statistics
        """
        query1 = """
        SELECT ip.part_num, SUM(ip.quantity) as quantity
        FROM inventory_parts ip
        JOIN inventories i ON ip.inventory_id = i.id
        WHERE i.set_num = ? AND ip.is_spare = 'f'
        GROUP BY ip.part_num
        """
        
        query2 = """
        SELECT ip.part_num, SUM(ip.quantity) as quantity
        FROM inventory_parts ip
        JOIN inventories i ON ip.inventory_id = i.id
        WHERE i.set_num = ? AND ip.is_spare = 'f'
        GROUP BY ip.part_num
        """
        
        set1_parts = self.query_to_dataframe(query1, (set_num1,))
        set2_parts = self.query_to_dataframe(query2, (set_num2,))
        
        if set1_parts.empty or set2_parts.empty:
            return {
                "overlap_percent_set1": 0,
                "overlap_percent_set2": 0,
                "common_parts": 0,
                "total_parts_set1": len(set1_parts),
                "total_parts_set2": len(set2_parts)
            }
        
        # Find common parts
        common_parts = pd.merge(set1_parts, set2_parts, on='part_num')
        
        # Calculate statistics
        total_parts_set1 = len(set1_parts)
        total_parts_set2 = len(set2_parts)
        common_part_count = len(common_parts)
        
        overlap_percent_set1 = (common_part_count / total_parts_set1) * 100 if total_parts_set1 > 0 else 0
        overlap_percent_set2 = (common_part_count / total_parts_set2) * 100 if total_parts_set2 > 0 else 0
        
        return {
            "overlap_percent_set1": overlap_percent_set1,
            "overlap_percent_set2": overlap_percent_set2,
            "common_parts": common_part_count,
            "total_parts_set1": total_parts_set1,
            "total_parts_set2": total_parts_set2
        }
    
    def get_all_theme_names(self):
        """
        Get mapping of theme IDs to names
        
        Returns:
            dict: Mapping from theme_id to theme_name
        """
        query = """
        SELECT id, name, parent_id
        FROM themes
        ORDER BY name
        """
        
        theme_df = self.query_to_dataframe(query)
        
        if theme_df.empty:
            return {}
            
        # Create mapping dictionary
        theme_map = {row['id']: row['name'] for _, row in theme_df.iterrows()}
        
        return theme_map