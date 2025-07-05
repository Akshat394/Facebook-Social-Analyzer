#!/usr/bin/env python3
"""
Facebook Ads Analytics Platform
Production-ready main application
"""

import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import numpy as np

from facebook_api import FacebookAPI
from database import DatabaseManager
from gemini_query import GeminiQueryEngine
from utils import format_currency, format_percentage, validate_account_id

# Page configuration
st.set_page_config(
    page_title="Facebook Ads Analytics",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'data_fetched' not in st.session_state:
    st.session_state.data_fetched = False
if 'campaigns_df' not in st.session_state:
    st.session_state.campaigns_df = None
if 'adsets_df' not in st.session_state:
    st.session_state.adsets_df = None
if 'ads_df' not in st.session_state:
    st.session_state.ads_df = None

def safe_convert_to_numeric(value, default=0):
    """Safely convert a value to numeric, handling strings and None values."""
    try:
        if pd.isna(value) or value is None or value == '':
            return default
        return float(value)
    except (ValueError, TypeError):
        return default

def prepare_numeric_data(df, columns):
    """Convert specified columns to numeric values safely."""
    df_copy = df.copy()
    for col in columns:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].apply(lambda x: safe_convert_to_numeric(x, 0))
    return df_copy

def main():
    st.title("📊 Facebook Ads Analytics Platform")
    st.markdown("### AI-Powered Facebook Ads Data Analysis")

    try:
        db_manager = DatabaseManager()
        facebook_api = FacebookAPI()
        
        # Initialize Gemini engine with error handling
        try:
            gemini_engine = GeminiQueryEngine(db_manager)
        except Exception as gemini_error:
            st.warning(f"Gemini AI features may not work: {str(gemini_error)}")
            gemini_engine = None
            
    except Exception as e:
        st.error(f"Failed to initialize components: {str(e)}")
        st.stop()

    with st.sidebar:
        st.header("🔧 Configuration")

        account_id = st.text_input(
            "Facebook Ads Account ID",
            placeholder="act_1234567890",
            help="Enter your Facebook Ads account ID (e.g., act_1234567890)"
        )

        if account_id and not validate_account_id(account_id):
            st.error("Invalid account ID format. Should start with 'act_' followed by numbers.")

        st.subheader("📅 Data Management")

        fetch_button = st.button(
            "🔄 Fetch Fresh Data",
            disabled=not account_id or not validate_account_id(account_id),
            help="Fetch latest data from Facebook Marketing API"
        )

        if fetch_button:
            fetch_facebook_data(account_id, facebook_api, db_manager)

        load_button = st.button(
            "📂 Load Existing Data",
            help="Load previously fetched data from database"
        )

        if load_button:
            load_existing_data(db_manager)

        if st.session_state.data_fetched:
            st.success("✅ Data loaded successfully")
            if st.session_state.campaigns_df is not None:
                st.metric("Campaigns", len(st.session_state.campaigns_df))
            if st.session_state.adsets_df is not None:
                st.metric("Ad Sets", len(st.session_state.adsets_df))
            if st.session_state.ads_df is not None:
                st.metric("Ads", len(st.session_state.ads_df))
        else:
            st.info("💡 Fetch or load data to begin analysis")

    if not st.session_state.data_fetched:
        show_welcome_screen()
    else:
        show_analytics_dashboard(gemini_engine)

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_facebook_data(account_id, facebook_api, db_manager):
    with st.spinner("Fetching data from Facebook Marketing API..."):
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            status_text.text("Fetching campaigns...")
            campaigns = facebook_api.fetch_campaigns(account_id)
            progress_bar.progress(25)

            status_text.text("Fetching ad sets...")
            adsets = facebook_api.fetch_adsets(account_id)
            progress_bar.progress(50)

            status_text.text("Fetching ads...")
            ads = facebook_api.fetch_ads(account_id)
            progress_bar.progress(75)

            status_text.text("Storing data in database...")
            db_manager.store_campaigns(campaigns)
            db_manager.store_adsets(adsets)
            db_manager.store_ads(ads)

            st.session_state.campaigns_df = pd.DataFrame(campaigns)
            st.session_state.adsets_df = pd.DataFrame(adsets)
            st.session_state.ads_df = pd.DataFrame(ads)
            st.session_state.data_fetched = True

            progress_bar.progress(100)
            status_text.text("✅ Data fetched and stored successfully!")

            time.sleep(1)
            st.rerun()

        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            progress_bar.empty()
            status_text.empty()

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def load_existing_data(db_manager):
    try:
        with st.spinner("Loading data from database..."):
            campaigns = db_manager.get_campaigns()
            adsets = db_manager.get_adsets()
            ads = db_manager.get_ads()

            if campaigns:
                st.session_state.campaigns_df = pd.DataFrame(campaigns)
                st.session_state.adsets_df = pd.DataFrame(adsets)
                st.session_state.ads_df = pd.DataFrame(ads)
                st.session_state.data_fetched = True
                st.success("Data loaded successfully from database!")
                st.rerun()
            else:
                st.warning("No data found in database. Please fetch fresh data first.")

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")

def show_analytics_dashboard(gemini_engine):
    """Show the main analytics dashboard"""
    tabs = st.tabs(["📊 Overview", "🧠 AI Query", "📋 Raw Data"])
    
    with tabs[0]:
        show_overview_tab()
    
    with tabs[1]:
        show_ai_query_tab(gemini_engine)
    
    with tabs[2]:
        show_raw_data_tab()

def show_overview_tab():
    st.header("📊 Campaign Overview")

    if st.session_state.ads_df is not None and not st.session_state.ads_df.empty:
        # Check which columns exist and prepare data accordingly
        available_columns = st.session_state.ads_df.columns.tolist()
        numeric_columns = []
        
        # Check for common column names
        spend_col = None
        impressions_col = None
        clicks_col = None
        campaign_col = None
        
        # Look for spend-related columns
        for col in available_columns:
            if 'spend' in col.lower():
                spend_col = col
                numeric_columns.append(col)
            elif 'impression' in col.lower():
                impressions_col = col
                numeric_columns.append(col)
            elif 'click' in col.lower():
                clicks_col = col
                numeric_columns.append(col)
            elif 'campaign' in col.lower() and 'name' in col.lower():
                campaign_col = col
        
        # Prepare numeric data for available columns
        insights_df = prepare_numeric_data(st.session_state.ads_df, numeric_columns)
        
        # Display metrics for available columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if spend_col:
                total_spend = insights_df[spend_col].sum()
                st.metric("Total Spend", format_currency(total_spend))
            else:
                st.metric("Total Spend", "N/A")
        
        with col2:
            if impressions_col:
                total_impressions = insights_df[impressions_col].sum()
                st.metric("Total Impressions", f"{total_impressions:,}")
            else:
                st.metric("Total Impressions", "N/A")
        
        with col3:
            if clicks_col:
                total_clicks = insights_df[clicks_col].sum()
                st.metric("Total Clicks", f"{total_clicks:,}")
            else:
                st.metric("Total Clicks", "N/A")
        
        with col4:
            if impressions_col and clicks_col:
                total_impressions = insights_df[impressions_col].sum()
                total_clicks = insights_df[clicks_col].sum()
                avg_ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
                st.metric("Average CTR", format_percentage(avg_ctr))
            else:
                st.metric("Average CTR", "N/A")
        
        # Display charts if we have the necessary data
        if spend_col or campaign_col:
            col1, col2 = st.columns(2)
            
            with col1:
                if spend_col and campaign_col:
                    spend_by_campaign = insights_df.groupby(campaign_col)[spend_col].sum().sort_values(ascending=False)
                    fig_spend = px.bar(
                        x=spend_by_campaign.values,
                        y=spend_by_campaign.index,
                        orientation='h',
                        title="Spend by Campaign",
                        labels={'x': 'Spend ($)', 'y': 'Campaign'}
                    )
                    st.plotly_chart(fig_spend, use_container_width=True)
                else:
                    st.info("Spend and campaign data not available for visualization")
            
            with col2:
                if clicks_col and impressions_col and campaign_col:
                    ctr_by_campaign = insights_df.groupby(campaign_col).apply(
                        lambda x: (x[clicks_col].sum() / x[impressions_col].sum() * 100) if x[impressions_col].sum() > 0 else 0
                    ).sort_values(ascending=False)
                    fig_ctr = px.bar(
                        x=ctr_by_campaign.values,
                        y=ctr_by_campaign.index,
                        orientation='h',
                        title="CTR by Campaign",
                        labels={'x': 'CTR (%)', 'y': 'Campaign'}
                    )
                    st.plotly_chart(fig_ctr, use_container_width=True)
                else:
                    st.info("CTR data not available for visualization")
        
        # Show available columns for debugging
        with st.expander("🔍 Available Data Columns"):
            st.write("Columns in ads data:")
            st.write(available_columns)
            if st.session_state.campaigns_df is not None:
                st.write("Columns in campaigns data:")
                st.write(st.session_state.campaigns_df.columns.tolist())

def show_ai_query_tab(gemini_engine):
    st.header("🧠 AI-Powered Query")
    
    if gemini_engine is None:
        st.error("❌ Gemini AI engine is not available. Please check your GEMINI_API_KEY environment variable.")
        st.info("You can still view your data in the Overview and Raw Data tabs.")
        return
    
    st.write("Ask questions about your Facebook Ads data in natural language.")
    
    # Example queries
    st.subheader("💡 Example Queries:")
    example_queries = [
        "How many campaigns do I have?",
        "Show me active campaigns",
        "Campaign status overview",
        "Budget analysis",
        "Recent campaigns and ads"
    ]
    
    # Create columns for example queries
    cols = st.columns(len(example_queries))
    for i, query in enumerate(example_queries):
        with cols[i]:
            if st.button(query, key=f"example_{i}"):
                st.session_state.user_query = query
    
    # User input
    user_query = st.text_area(
        "Ask your question:",
        value=st.session_state.get('user_query', ''),
        placeholder="e.g., 'Which campaigns have the highest ROI?', 'Show me spending trends over time'",
        height=100
    )
    
    if st.button("🔍 Analyze", type="primary"):
        if user_query.strip():
            with st.spinner("Analyzing your data..."):
                try:
                    # Check if we have data to analyze
                    if st.session_state.ads_df is not None and not st.session_state.ads_df.empty:
                        result = gemini_engine.query(user_query)
                        st.success("Analysis complete!")
                        st.markdown(result)
                    else:
                        st.error("No data available for analysis. Please fetch data first.")
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    st.info("Make sure your Gemini API key is properly configured.")
        else:
            st.warning("Please enter a question to analyze.")
    
    # Show data preview for context
    if st.session_state.ads_df is not None and not st.session_state.ads_df.empty:
        with st.expander("📋 Data Preview for AI Analysis"):
            st.write("Available data for AI analysis:")
            st.dataframe(st.session_state.ads_df.head())
            
            # Show column information
            st.write("**Available columns:**")
            for col in st.session_state.ads_df.columns:
                st.write(f"- {col}")

def show_raw_data_tab():
    st.header("📋 Raw Data")
    
    if st.session_state.campaigns_df is not None:
        st.subheader("Campaigns")
        st.dataframe(st.session_state.campaigns_df, use_container_width=True)
    
    if st.session_state.adsets_df is not None:
        st.subheader("Ad Sets")
        st.dataframe(st.session_state.adsets_df, use_container_width=True)
    
    if st.session_state.ads_df is not None:
        st.subheader("Ads")
        st.dataframe(st.session_state.ads_df, use_container_width=True)

def show_welcome_screen():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        ## 🚀 Welcome to Facebook Ads Analytics

        Get started by:
        1. **Enter your Facebook Ads Account ID** in the sidebar
        2. **Fetch fresh data** from Facebook Marketing API, or
        3. **Load existing data** from the database

        Once data is loaded, you can:
        - 📊 View comprehensive analytics dashboards
        - 🧠 Ask natural language questions about your data
        - 📈 Explore interactive charts and visualizations

        ### Features:
        - 🔄 Real-time data fetching with pagination
        - 📀 Database storage
        - 🧠 AI-powered querying with Google Gemini
        - 📱 Interactive dashboards and charts
        """)

if __name__ == "__main__":
    main() 