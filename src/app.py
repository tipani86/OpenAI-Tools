import os
import base64
import asyncio
import argparse
import pandas as pd
import streamlit as st
from pathlib import Path
from loguru import logger
from openai_utils import *
from datetime import datetime, timedelta

UTC_TIMESTAMP = int(datetime.utcnow().timestamp())

FILE_ROOT = Path(__file__).resolve().parent

parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true", help="Enable debug mode")
args = parser.parse_args()

DEBUG = args.debug

@st.cache_data(show_spinner=False)
def get_css() -> str:
    # Read CSS code from style.css file
    with open(FILE_ROOT / "style.css", "r") as f:
        return f"<style>{f.read()}</style>"
    
def df_to_csv(df: pd.DataFrame) -> str:
    return base64.b64encode(df.to_csv().encode("utf-8")).decode()

def get_action_names(key: str) -> str:
    mapping = {
        "review": "Review File",
        "delete": "Delete File"
    }
    return mapping[key]

def is_num(inp) -> bool:
    try:
        inp = int(inp)
        return True
    except:
        return False

### MAIN APP STARTS HERE ###

# Define overall layout
st.set_page_config(
    page_title="OpenAI Tools",
    page_icon="https://openaiapi-site.azureedge.net/public-assets/d/e90ac348d3/favicon.png",
    initial_sidebar_state=st.session_state.get('sidebar_state', 'expanded'),
    layout="wide"
)

st.session_state["openai_api_key"] = os.getenv("OPENAI_API_KEY", "")
st.session_state["openai_org_id"] = os.getenv("OPENAI_ORGANIZATION", "")

# Load CSS code
st.markdown(get_css(), unsafe_allow_html=True)

# Define sidebar layout
with st.sidebar:
    st.subheader("OpenAI Credentials")
    st.text_input(label="openai_api_key", key="openai_api_key", placeholder="Your OpenAI API Key", label_visibility="collapsed")
    st.text_input(label="openai_org_id", key="openai_org_id", placeholder="Your OpenAI Organization ID", label_visibility="collapsed")
    st.caption("_**Author's Note:** While I can only claim that your credentials are not stored anywhere, for maximum security, you should generate a new app-specific API key on your OpenAI account page and use it here. That way, you can deactivate that key after you don't plan to use this app anymore, and it won't affect any of your other apps. You can check out the GitHub source for this app using below button:_")
    st.markdown('<a href="https://github.com/tipani86/OpenAI-Tools"><img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/tipani86/OpenAI-Tools?style=social"></a>', unsafe_allow_html=True)
    if DEBUG:
        if st.button("Reload page"):
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

async def get_files() -> list:
    return await openai_tool_op.get_files(openai_api_key, openai_org_id)

async def view_file_contents(file_id: str) -> list:
    return await openai_tool_op.view_file_contents(openai_api_key, openai_org_id, file_id)

async def delete_file(file_id: str) -> dict:
    return await openai_tool_op.delete_file(openai_api_key, openai_org_id, file_id)

# Define main layout

async def main():
    with st.spinner("Loading data..."):
        users_res = await get_users()
        files_res = await get_files()

    with st.expander("**My Organization's Users**", expanded=True):
        users_df = pd.DataFrame(users_res["data"])
        # Convert "created" column to datetime
        users_df["created"] = pd.to_datetime(users_df["created"], unit="s")
        st.dataframe(users_df[["created", "role", "name"]].set_index("created", inplace=False))

    with st.expander("**Token Usage**", expanded=False):
        def get_user_name(id: str) -> str:
            return users_df[users_df["id"] == id]["name"].values[0]
        with st.form("usage_form"):
            user_selector = st.selectbox("Select a user", users_df["id"].values, format_func=get_user_name)
            start_date_col, end_date_col = st.columns(2)
            with start_date_col:
                # Default value is today minus 7 days, output should be converted to "YYYY-MM-DD" string
                start_date = st.date_input("Start date", value=datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
            with end_date_col:
                # Default value is today (minus 1 day), output should be converted to "YYYY-MM-DD" string
                end_date = st.date_input("End date", value=datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
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

    if not DEBUG:
        st.stop()

    with st.expander("**Model Fine-Tuning**", expanded=False):
        upload_column, files_column = st.columns(2)

        with files_column:
            st.caption("Uploaded Files")
            if len(files_res["data"]) == 0:
                st.text("No files found.")
            else:
                files_df = pd.DataFrame(files_res["data"])
                files_df = files_df[["created_at", "id", "filename", "bytes", "purpose"]].set_index("created_at")
                files_df.index = pd.to_datetime(files_df.index, unit="s")
                st.dataframe(files_df)

        with upload_column:
            st.caption("Upload `JsonLines` File(s)")
            with st.form("upload_form", clear_on_submit=True):
                upload_files = st.file_uploader("Upload", label_visibility="collapsed", type="jsonl", accept_multiple_files=True, key="upload_files")
                upload_submitted = st.form_submit_button("Check File(s) and Upload")
            if upload_submitted:
                if len(upload_files) == 0:
                    st.warning("No files selected.")
                else:
                    errors = 0
                    for file in upload_files:
                        st.write(f"`{file.name}`")
                        check_res = check_finetune_dataset(file)
                        if check_res["format_errors"]:
                            error_msg = f"File `{file.name}` has the following errors:\n"
                            for error in check_res["format_errors"]:
                                error_msg += f"- {error}: {check_res['format_errors'][error]}\n"
                            st.error(error_msg)
                            errors += 1
                        else:
                            st.caption("File is valid.")
                    if errors == 0:
                        with st.spinner("Uploading files..."):
                            for file in stqdm(upload_files):
                                await openai_tool_op.upload_file(file)
                        st.success("All files uploaded successfully.")
                        if files_column.button("Refresh file list"):
                            st.experimental_rerun()
                    else:
                        st.error(f"{errors} file(s) had errors. Please fix them and try again.")
        if len(files_res["data"]) > 0:
            with st.form("file_operation_form", clear_on_submit=True):
                file_id_col, epochs_col, action_col = st.columns([5, 2, 3])
                with file_id_col:
                    file_id = st.text_input("File ID", placeholder="Paste file ID from above list")
                with epochs_col:
                    epochs = st.text_input("Target Epochs", value=3)
                with action_col:
                    action = st.selectbox(
                        label="Action",
                        options=("review", "delete"),
                        format_func=get_action_names,
                    )
                file_op_submitted = st.form_submit_button("Submit")
            
            if file_op_submitted:
                if file_id not in files_df.id.tolist():
                    st.error(f"File ID `{file_id}` not found.")
                elif not is_num(epochs) or int(epochs) < 1 or int(epochs) > 25:
                    st.error(f"Invalid value {epochs} for `epochs` (1 <= `epochs` <= 25)")
                elif action == "review":
                    epochs = int(epochs)
                    contents_res = await view_file_contents(file_id)
                    lines = contents_res.decode("utf-8").split("\n")
                    dataset = [json.loads(line) for line in lines]
                    st.write(f"`{file_id}`")
                    st.caption("Contents")
                    st.json(dataset, expanded=False)
                    check_res = check_finetune_dataset(dataset)
                    if check_res["format_errors"]:
                        error_msg = f"File `{file.name}` has the following errors:\n"
                        for error in check_res["format_errors"]:
                            error_msg += f"- {error}: {check_res['format_errors'][error]}\n"
                        st.error(error_msg)
                    else:
                        tokens_estimate = estimate_training_tokens(
                            dataset = check_res["dataset"],
                            convo_lens = check_res["convo_lens"],
                            epochs=epochs
                        )
                        st.info(
                            f"Review passed. Dataset includes ~{tokens_estimate['n_billing_tokens']} tokens. By default, you'll train for {tokens_estimate['n_epochs']} epochs on this dataset. "
                            f"It amounts to ~{tokens_estimate['n_billing_tokens'] * tokens_estimate['n_epochs']} tokens in total. Check OpenAI pricing page to estimate total costs.",
                            icon="✅"
                        )
                        with st.form("finetune-form", clear_on_submit=True):
                            st.caption("**Train a Finetuned Model**")
                            train_id_col, epochs_col2 = st.columns(2)
                            with train_id_col:
                                st.text_input("Training File ID (Review another File ID to change)", value=file_id, disabled=True)
                            with epochs_col2:
                                epochs = st.number_input(
                                    "Actual Epochs to Train",
                                    value=tokens_estimate["n_epochs"], min_value=1, max_value=25, step=1
                                )
                            val_id_col, suffix_col = st.columns(2)
                            with val_id_col:
                                validation_id = st.text_input(
                                    "Validation File ID (Optional)",
                                    placeholder="Enter File ID for a validation dataset",
                                    help="If you provide this file, the data is used to generate validation metrics periodically during fine-tuning. These metrics can be viewed in the fine-tuning results file. The same data should not be present in both train and validation files."
                                )
                            with suffix_col:
                                suffix = st.text_input(
                                    "Custom Model Name (Optional)", 
                                    max_chars=40,
                                    help='A string of up to 40 characters that will be added to your fine-tuned model name. For example, "custom-model-name" would produce a model name like `ft:gpt-3.5-turbo:openai:custom-model-name:7p4lURel`.'
                                )
                            finetune_submitted = st.form_submit_button("Start Finetuning Job")

                elif action == "delete":
                    delete_res = await delete_file(file_id)
                    if not delete_res:
                        st.error(f"File {file_id} deletion failed")
                        if not delete_res["deleted"]:
                            st.json(delete_res)
                    else:
                        st.experimental_rerun()

asyncio.run(main())