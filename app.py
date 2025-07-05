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
        gemini_engine = GeminiQueryEngine(db_manager)
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
        # Prepare data with proper numeric conversion
        insights_df = prepare_numeric_data(st.session_state.ads_df, 
                                         ['spend', 'impressions', 'clicks'])

        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_spend = insights_df['spend'].sum()
            st.metric("Total Spend", format_currency(total_spend))
        
        with col2:
            total_impressions = insights_df['impressions'].sum()
            st.metric("Total Impressions", f"{total_impressions:,}")
        
        with col3:
            total_clicks = insights_df['clicks'].sum()
            st.metric("Total Clicks", f"{total_clicks:,}")
        
        with col4:
            avg_ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
            st.metric("Average CTR", format_percentage(avg_ctr))

        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Spend by campaign
            if 'campaign_name' in insights_df.columns:
                spend_by_campaign = insights_df.groupby('campaign_name')['spend'].sum().sort_values(ascending=False)
                fig_spend = px.bar(
                    x=spend_by_campaign.values,
                    y=spend_by_campaign.index,
                    orientation='h',
                    title="Spend by Campaign",
                    labels={'x': 'Spend ($)', 'y': 'Campaign'}
                )
                st.plotly_chart(fig_spend, use_container_width=True)
        
        with col2:
            # CTR by campaign
            if 'campaign_name' in insights_df.columns:
                ctr_by_campaign = insights_df.groupby('campaign_name').apply(
                    lambda x: (x['clicks'].sum() / x['impressions'].sum() * 100) if x['impressions'].sum() > 0 else 0
                ).sort_values(ascending=False)
                fig_ctr = px.bar(
                    x=ctr_by_campaign.values,
                    y=ctr_by_campaign.index,
                    orientation='h',
                    title="CTR by Campaign",
                    labels={'x': 'CTR (%)', 'y': 'Campaign'}
                )
                st.plotly_chart(fig_ctr, use_container_width=True)

def show_ai_query_tab(gemini_engine):
    st.header("🧠 AI-Powered Query")
    
    user_query = st.text_area(
        "Ask a question about your Facebook Ads data:",
        placeholder="e.g., 'Which campaigns have the highest ROI?', 'Show me spending trends over time'",
        height=100
    )
    
    if st.button("🔍 Analyze", type="primary"):
        if user_query.strip():
            with st.spinner("Analyzing your data..."):
                try:
                    result = gemini_engine.query(user_query)
                    st.success("Analysis complete!")
                    st.markdown(result)
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
        else:
            st.warning("Please enter a question to analyze.")

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