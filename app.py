import streamlit as st
import asyncio
import os
import json
from datetime import datetime, date, timedelta, timezone
import pandas as pd
import traceback
from io import BytesIO

from stqdm import stqdm
from api_client import OpenAIUsageAPIClient
from data_processor import create_dataframe, enrich_dataframe_with_lookups, reorder_columns

# API endpoints configuration
API_ENDPOINTS = {
    "completions": "https://api.openai.com/v1/organization/usage/completions",
    "embeddings": "https://api.openai.com/v1/organization/usage/embeddings", 
    "moderations": "https://api.openai.com/v1/organization/usage/moderations",
    "images": "https://api.openai.com/v1/organization/usage/images",
    "audio_speeches": "https://api.openai.com/v1/organization/usage/audio_speeches",
    "audio_transcriptions": "https://api.openai.com/v1/organization/usage/audio_transcriptions",
    "vector_stores": "https://api.openai.com/v1/organization/usage/vector_stores",
    "code_interpreter_sessions": "https://api.openai.com/v1/organization/usage/code_interpreter_sessions"
}

# Lookup endpoints for ID-to-name mapping
LOOKUP_ENDPOINTS = {
    "users": "https://api.openai.com/v1/organization/users",
    "projects": "https://api.openai.com/v1/organization/projects",
    "api_keys": "https://api.openai.com/v1/organization/admin_api_keys"
}


def date_to_unix_timestamp(date_obj: date) -> int:
    """Convert date to Unix timestamp, treating the date as UTC"""
    # Create a datetime object and explicitly set timezone to UTC
    dt_utc = datetime.combine(date_obj, datetime.min.time(), timezone.utc)
    return int(dt_utc.timestamp())


async def main_async():
    # Page configuration
    st.set_page_config(
        page_title="OpenAI Usage API Dashboard",
        page_icon="📊",
        layout="wide"
    )
    
    # Sidebar with API key
    st.sidebar.header("Configuration")
    
    # Check environment for API key and pre-populate
    env_api_key = os.getenv("OPENAI_ADMIN_KEY", "")
    api_key = st.sidebar.text_input(
        "OpenAI Admin API Key",
        value=env_api_key,
        type="password",
        help="Enter your [OpenAI Admin API key](https://platform.openai.com/settings/organization/admin-keys). If OPENAI_ADMIN_KEY environment variable is set, it will be pre-populated."
    )
    
    # Author's security note
    st.sidebar.markdown("""
    _**Author's Note:** While I can only claim that your credentials are not stored anywhere, for maximum security, you should generate a new app-specific and read only [Admin API key](https://platform.openai.com/settings/organization/admin-keys) on your account and use it here. This way, you can deactivate the key after you don't plan to use the app anymore, and it won't affect any of your other keys/apps. You can check out the GitHub source for this app using below button:_
    """)
    
    # GitHub repository badge
    st.sidebar.markdown("""
    [![GitHub](https://img.shields.io/github/stars/tipani86/OpenAI-Tools)](https://github.com/tipani86/OpenAI-Tools)
    """)
    
    # Cache management
    st.sidebar.header("Cache Management")
    if st.sidebar.button("🗑️ Clear All Cache", help="Clear cached API responses and session data"):
        # Clear the alru cache if client exists
        if 'api_client' in st.session_state:
            st.session_state.api_client.fetch_all_pages.cache_clear()
        # Clear session state
        st.session_state.clear()
        st.sidebar.success("Cache and session data cleared successfully!")

    # Main content
    st.title("📊 OpenAI Usage API Dashboard")
    st.markdown("Fetch and analyze usage data from OpenAI's Usage API across all endpoints.")
    
    # Form for user inputs
    with st.form("usage_form"):
        st.subheader("Settings")
        
        # Date range picker
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date (inclusive)",
                value=date.today() - timedelta(days=7),
                help="Start date for usage data (inclusive)"
            )
        with col2:
            end_date = st.date_input(
                "End Date (**exclusive**; for monthly reports, select the first day of the _second_ month)", 
                value=date.today(),
                help="End date for usage data (exclusive)"
            )
        
        # Usage endpoints selection
        selected_endpoints = st.multiselect(
            "Usage Endpoints",
            options=list(API_ENDPOINTS.keys()),
            default=list(API_ENDPOINTS.keys()),
            help="Select which usage endpoints to query"
        )
        
        # Submit button
        submit_button = st.form_submit_button("Fetch Usage Data", type="primary")
    
    # Process form submission
    if submit_button:
        if not api_key:
            st.error("Please provide an OpenAI Admin API key.")
            return
            
        if not selected_endpoints:
            st.error("Please select at least one usage endpoint.")
            return
            
        try:
            # Convert dates to Unix timestamps
            start_timestamp = date_to_unix_timestamp(start_date)
            end_timestamp = date_to_unix_timestamp(end_date)
            
            # Create or reuse API client from session state
            if 'api_client' not in st.session_state or st.session_state.get('api_key') != api_key:
                # Close existing client if it exists
                if 'api_client' in st.session_state:
                    await st.session_state.api_client.close()
                # Create new client
                st.session_state.api_client = OpenAIUsageAPIClient(api_key)
                st.session_state.api_key = api_key
            
            client = st.session_state.api_client
            
            # Prepare parameters
            base_params = {
                "start_time": start_timestamp,
                "end_time": end_timestamp,
                "group_by": ["project_id", "user_id", "api_key_id", "model"]
            }
            
            # Create tasks for asyncio.gather
            tasks = []
            for endpoint_name in selected_endpoints:
                endpoint_url = API_ENDPOINTS[endpoint_name]
                params = base_params.copy()
                
                # Special provision for vector stores - can only be grouped by project_id
                if endpoint_name == "vector_stores":
                    params["group_by"] = ["project_id"]
                # Code interpreter sessions also can only be grouped by project_id
                elif endpoint_name == "code_interpreter_sessions":
                    params["group_by"] = ["project_id"]
                
                # Convert params dict to JSON string for caching
                params_json = json.dumps(params, sort_keys=True)
                task = client.fetch_all_pages(endpoint_name, endpoint_url, params_json)
                tasks.append(task)
            
            # Execute tasks with progress tracking
            st.info(f"Fetching data from {len(tasks)} endpoints...")
            
            all_data = await stqdm.gather(*tasks)

            # all_data is a list of lists, each list contains the data for a single endpoint, need to flatten it
            all_data = [item for sublist in all_data for item in sublist]
            
            # Fetch lookup data for enrichment
            st.info("Fetching lookup data for users, projects, and API keys...")
            lookup_tasks = [
                client.fetch_lookup_data("users", LOOKUP_ENDPOINTS["users"]),
                client.fetch_lookup_data("projects", LOOKUP_ENDPOINTS["projects"]),
                client.fetch_lookup_data("api_keys", LOOKUP_ENDPOINTS["api_keys"])
            ]
            user_lookup, project_lookup, api_key_lookup = await stqdm.gather(*lookup_tasks)
            
            # Display cache statistics only in debug mode
            if os.getenv("LOGURU_LEVEL") == "DEBUG":
                cache_info = client.fetch_all_pages.cache_info()
                st.info(f"📊 Cache Statistics: {cache_info.hits} hits, {cache_info.misses} misses, {cache_info.currsize}/{cache_info.maxsize} cached entries")
            
            # Process and display results
            if all_data:
                df = create_dataframe(all_data)
                # Enrich dataframe with human-readable names
                df = enrich_dataframe_with_lookups(df, user_lookup, project_lookup, api_key_lookup)
                # Reorder columns for better presentation
                df = reorder_columns(df)
                
                st.success(f"✅ Successfully fetched {len(df)} records from {len(selected_endpoints)} endpoints")
                
                # Display DataFrame
                st.subheader("Usage Data")
                st.dataframe(df, use_container_width=True)
                
                # Download functionality
                def convert_df_to_csv(dataframe):
                    return dataframe.to_csv(index=False)
                
                csv_data = convert_df_to_csv(df)
                
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv_data,
                    file_name=f"openai_usage_{start_date}_{end_date}.csv",
                    mime="text/csv"
                )
                
            else:
                st.warning("No data found for the selected parameters and date range.")
                
        except Exception as e:
            st.error("An error occurred while fetching usage data:")
            st.error(f"Error: {str(e)}")
            st.error("Full traceback:")
            st.code(traceback.format_exc())


if __name__ == "__main__":
    asyncio.run(main_async()) 