import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64

# Removed sample parts function to focus on real data only

def create_donut_chart(values, labels, title):
    """
    Create a donut chart
    
    Args:
        values (list): List of values
        labels (list): List of labels
        title (str): Chart title
        
    Returns:
        plotly.graph_objects.Figure: Donut chart
    """
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.4,
        textinfo='label+percent'
    )])
    
    fig.update_layout(
        title=title,
        height=400,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    return fig

def create_bar_chart(data, x, y, title, color=None):
    """
    Create a bar chart
    
    Args:
        data (DataFrame): DataFrame with data
        x (str): Column for x-axis
        y (str): Column for y-axis
        title (str): Chart title
        color (str): Column for color
        
    Returns:
        plotly.express.Figure: Bar chart
    """
    fig = px.bar(
        data,
        x=x,
        y=y,
        color=color,
        title=title,
        height=400
    )
    
    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    return fig

def create_heatmap(data, title):
    """
    Create a heatmap
    
    Args:
        data (DataFrame): DataFrame with data
        title (str): Chart title
        
    Returns:
        plotly.graph_objects.Figure: Heatmap
    """
    fig = px.imshow(
        data,
        labels=dict(x="Set", y="Set", color="Overlap %"),
        title=title,
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        height=600,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    return fig

def get_theme_color_map(theme_ids):
    """
    Generate a color map for themes
    
    Args:
        theme_ids (list): List of theme IDs
        
    Returns:
        dict: Mapping of theme_id to color
    """
    colors = px.colors.qualitative.Plotly
    return {theme_id: colors[i % len(colors)] for i, theme_id in enumerate(theme_ids)}

def export_to_csv(df, filename):
    """
    Generate a download link for a DataFrame as CSV
    
    Args:
        df (DataFrame): DataFrame to export
        filename (str): Filename for the CSV
        
    Returns:
        str: HTML for download link
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

def format_part_list(parts_df):
    """
    Format a parts list for display
    
    Args:
        parts_df (DataFrame): DataFrame with parts
        
    Returns:
        DataFrame: Formatted parts DataFrame
    """
    if parts_df.empty:
        return parts_df
        
    # Select relevant columns
    if 'name' in parts_df.columns:
        display_df = parts_df[['part_num', 'name', 'quantity']].copy()
    else:
        display_df = parts_df[['part_num', 'quantity']].copy()
        
    return display_df
