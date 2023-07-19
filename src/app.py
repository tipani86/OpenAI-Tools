import base64
import asyncio
import pandas as pd
import streamlit as st
from pathlib import Path
from loguru import logger
from openai_utils import OpenAITools
from datetime import datetime, timedelta

FILE_ROOT = Path(__file__).resolve().parent

@st.cache_data(show_spinner=False)
def get_css() -> str:
    # Read CSS code from style.css file
    with open(FILE_ROOT / "style.css", "r") as f:
        return f"<style>{f.read()}</style>"
    
@st.cache_data
def df_to_csv(df: pd.DataFrame) -> str:
    return base64.b64encode(df.to_csv().encode("utf-8")).decode()

### MAIN APP STARTS HERE ###

# Define overall layout
st.set_page_config(
    page_title="OpenAI Tools",
    page_icon="https://openaiapi-site.azureedge.net/public-assets/d/e90ac348d3/favicon.png",
    initial_sidebar_state=st.session_state.get('sidebar_state', 'expanded')
)

# Load CSS code
st.markdown(get_css(), unsafe_allow_html=True)

# Define sidebar layout
with st.sidebar:
    st.subheader("OpenAI Credentials")
    st.text_input(label="openai_api_key", key="openai_api_key", placeholder="Your OpenAI API Key", label_visibility="collapsed")
    st.text_input(label="openai_org_id", key="openai_org_id", placeholder="Your OpenAI Organization ID", label_visibility="collapsed")
    st.caption("_**Author's Note:** While I can only claim that your credentials are not stored anywhere, for maximum security, you should generate a new app-specific API key on your OpenAI account page and use it here. That way, you can deactivate that key after you don't plan to use this app anymore, and it won't affect any of your other apps._")
    if st.button("Rerun"):
        st.experimental_rerun()

# Gate the loading of the rest of the page if the user hasn't entered their credentials
openai_api_key = st.session_state.get("openai_api_key", "")
openai_org_id = st.session_state.get("openai_org_id", "")

if len(openai_api_key) == 0 or len(openai_org_id) == 0:
    st.info("Please enter your OpenAI credentials in the sidebar to continue.")
    st.stop()

# Initialize OpenAI utils
openai_tool_op = OpenAITools()

async def get_users() -> list:
    return await openai_tool_op.get_users(openai_api_key, openai_org_id)

# Define main layout

async def main():
    st.subheader("My Organization")
    users_res = await get_users()
    users_df = pd.DataFrame(users_res["data"])
    # Convert "created" column to datetime
    users_df["created"] = pd.to_datetime(users_df["created"], unit="s")
    st.dataframe(users_df[["created", "role", "name"]].set_index("created", inplace=False))

    st.subheader("Token Usage")
    def get_user_name(id: str) -> str:
        return users_df[users_df["id"] == id]["name"].values[0]
    with st.form("usage_form"):
        user_selector = st.selectbox("Select a user", users_df["id"].values, format_func=get_user_name)
        start_date_col, end_date_col = st.columns(2)
        with start_date_col:
            # Default value is today minus 30 days, output should be converted to "YYYY-MM-DD" string
            start_date = st.date_input("Start date", value=datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        with end_date_col:
            # Default value is today, output should be converted to "YYYY-MM-DD" string
            end_date = st.date_input("End date", value=datetime.now()).strftime("%Y-%m-%d")
        usage_submit = st.form_submit_button("Generate Report")
    if usage_submit:
        # Check that end date is not before start date
        if end_date < start_date:
            st.error("End date cannot be before start date.")
            st.stop()
        with st.spinner("Loading usage data (this may take a while)..."):
            usage_res = await openai_tool_op.get_usage(openai_api_key, openai_org_id, user_selector, [start_date, end_date])
        usage_df = pd.DataFrame(usage_res["data"])
        # Convert "aggregation_timestamp" column to datetime that rounds to a single day only
        usage_df["date"] = pd.to_datetime(usage_df["aggregation_timestamp"], unit="s").dt.date
        # Rename column "snapshot_id" as "model"
        usage_df.rename(columns={"snapshot_id": "model"}, inplace=True)
        # Group by aggregation_timestamp and snapshot_id
        groups = usage_df.groupby(["date", "model"]).sum(numeric_only=True)
        # Clean up for display
        usage_df = groups.reset_index()
        usage_df = usage_df[["date", "model", "n_requests", "n_context_tokens_total", "n_generated_tokens_total"]].set_index("date", inplace=False)
        st.dataframe(usage_df)
        # Allow user to download usage data as CSV
        st.markdown(
            f'<a href="data:file/csv;base64,{df_to_csv(usage_df)}" download="usage_{start_date}_{end_date}.csv">Download usage data as CSV</a>',
            unsafe_allow_html=True
        )

asyncio.run(main())