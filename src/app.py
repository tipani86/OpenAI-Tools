import os
import openai
import base64
import asyncio
import argparse
import pandas as pd
import streamlit as st
from io import StringIO
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
def get_local_img(file_path: Path) -> str:
    # Load a byte image and return its base64 encoded string
    return base64.b64encode(open(file_path, "rb").read()).decode("utf-8")

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

openai_api_key = os.getenv("OPENAI_API_KEY", "")
openai_org_id = os.getenv("OPENAI_ORGANIZATION", "")
if len(openai_api_key) > 0:
    st.session_state["openai_api_key"] = openai_api_key
if len(openai_org_id) > 0:
    st.session_state["openai_org_id"] = openai_org_id

# Load CSS code
st.markdown(get_css(), unsafe_allow_html=True)

# Define sidebar layout
with st.sidebar:
    st.subheader("OpenAI Credentials")
    st.text_input(label="openai_api_key", key="openai_api_key", placeholder="Your OpenAI API Key", type="password", label_visibility="collapsed")
    st.text_input(label="openai_org_id", key="openai_org_id", placeholder="Your OpenAI Organization ID", type="password", label_visibility="collapsed")
    st.caption("_**Author's Note:** While I can only claim that your credentials are not stored anywhere, for maximum security, you should generate a new app-specific API key on your OpenAI account page and use it here. That way, you can deactivate that key after you don't plan to use this app anymore, and it won't affect any of your other apps. You can check out the GitHub source for this app using below button:_")
    st.markdown('<a href="https://github.com/tipani86/OpenAI-Tools"><img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/tipani86/OpenAI-Tools?style=social"></a><br><small>Page views: <img src="https://www.cutercounter.com/hits.php?id=hxncoqd&nd=4&style=1" border="0" alt="visitor counter"></small>', unsafe_allow_html=True)
    if DEBUG:
        if st.button("Reload page"):
            st.experimental_rerun()

# Gate the loading of the rest of the page if the user hasn't entered their credentials
openai_api_key = st.session_state.get("openai_api_key", "")
openai_org_id = st.session_state.get("openai_org_id", "")

openai.api_key = openai_api_key
openai.organization = openai_org_id

if len(openai_api_key) == 0 or len(openai_org_id) == 0:
    st.info("Please enter your OpenAI credentials in the sidebar to continue.")
    st.stop()

# Initialize OpenAI utils
openai_tool_op = OpenAITools()

async def get_users() -> dict:
    return await openai_tool_op.get_users(openai_api_key, openai_org_id)

async def get_files() -> dict:
    return await openai_tool_op.get_files(openai_api_key, openai_org_id)

async def upload_file(file):
    await openai_tool_op.upload_file(openai_api_key, openai_org_id, file)

async def view_file_contents(file_id: str) -> bytes:
    return await openai_tool_op.view_file_contents(openai_api_key, openai_org_id, file_id)

async def delete_file(file_id: str) -> dict | None:
    return await openai_tool_op.delete_file(openai_api_key, openai_org_id, file_id)

async def get_finetune_jobs() -> dict:
    return await openai_tool_op.get_finetune_jobs(openai_api_key, openai_org_id)

async def create_finetune_job(data: dict):
    await openai_tool_op.create_finetune_job(openai_api_key, openai_org_id, data)

async def cancel_finetune_job(job_id: str):
    await openai_tool_op.cancel_finetune_job(openai_api_key, openai_org_id, job_id)

async def get_models() -> dict:
    return await openai_tool_op.get_models(openai_api_key, openai_org_id)

# Define main layout

async def main():
    if "NEED_REFRESH" not in st.session_state or st.session_state["NEED_REFRESH"] is True:
        with st.spinner("Loading data..."):
            # Use async to group data loading tasks in parallel
            users_res, files_res, finetune_jobs_res, models_res = await asyncio.gather(
                get_users(), get_files(), get_finetune_jobs(), get_models()
            )
            st.session_state["DATA"] = {
                "users": users_res,
                "files": files_res,
                "finetune_jobs": finetune_jobs_res,
                "models": models_res
            }
            st.session_state["NEED_REFRESH"] = False
    else:
        users_res = st.session_state["DATA"]["users"]
        files_res = st.session_state["DATA"]["files"]
        finetune_jobs_res = st.session_state["DATA"]["finetune_jobs"]
        models_res = st.session_state["DATA"]["models"]

    with st.expander("**My Organization's Users**", expanded=True):
        users_df = pd.DataFrame(users_res["data"])
        users_df["created"] = pd.to_datetime(users_df["created"], unit="s")
        users_df = users_df.set_index("created")
        users_df.index.rename("created at (UTC)", inplace=True)
        st.dataframe(users_df[["role", "name"]].sort_index())

    with st.expander("**Token Usage**", expanded=True):
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
            st.balloons()
            if isinstance(usage_res, dict) and "error" in usage_res:
                st.error(f"{usage_res['error']['message']}")
            else:
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
                st.dataframe(usage_df.sort_index())
                # Allow user to download usage data as CSV
                st.markdown(
                    f'<a href="data:file/csv;base64,{df_to_csv(usage_df)}" download="usage_{start_date}_{end_date}.csv">Download usage data as CSV</a>',
                    unsafe_allow_html=True
                )

    with st.expander("**Model Fine-Tuning**", expanded=True):
        files_column, finetune_jobs_column = st.columns(2)

        with finetune_jobs_column:
            st.caption("Finetuning Jobs")
            if len(finetune_jobs_res["data"]) == 0:
                st.info("No finetuning jobs found. Start a new job by uploading/reviewing a dataset file below.")
            else:
                finetune_jobs_df = pd.DataFrame(finetune_jobs_res["data"])
                finetune_jobs_df["created_at"] = pd.to_datetime(finetune_jobs_df["created_at"], unit="s")
                finetune_jobs_df["finished_at"] = pd.to_datetime(finetune_jobs_df["finished_at"], unit="s")
                finetune_jobs_df = finetune_jobs_df.set_index("created_at")
                finetune_jobs_df.index.rename("created at (UTC)", inplace=True)
                st.dataframe(finetune_jobs_df[[
                    "id", "model", "status", "trained_tokens", "training_file", "validation_file", "hyperparameters", "finished_at", "fine_tuned_model", "result_files"
                ]].sort_index(), use_container_width=True)
                finetune_refresh_col, finetune_cancel_col = st.columns([1, 3])
                with finetune_refresh_col:
                    if st.button("Refresh List"):
                        st.session_state["NEED_REFRESH"] = True
                        st.experimental_rerun()
                with finetune_cancel_col:
                    with st.form("cancel_job_form", clear_on_submit=True):
                        job_id = st.text_input("Cancel Job :red[Immediately]", placeholder="Paste job ID from above list")
                        cancel_job_submitted = st.form_submit_button("Cancel Job")
                    if cancel_job_submitted and len(job_id) > 0:
                        cancel_res = await openai_tool_op.cancel_finetune_job(openai_api_key, openai_org_id, job_id)
                        if isinstance(cancel_res, dict) and "error" in cancel_res:
                            st.error(f"{cancel_res['error']['message']}")
                        else:
                            cancel_job_countdown = st.empty()
                            for i in range(5, 0, -1):
                                cancel_job_countdown.success(f"Cancellation submitted. Refreshing jobs list in {i} seconds...")
                                await asyncio.sleep(1)
                            st.session_state["NEED_REFRESH"] = True
                            st.experimental_rerun()

        with files_column:
            st.caption("Upload New `JsonLines` File(s)")
            with st.form("upload_form", clear_on_submit=True):
                upload_files = st.file_uploader("Upload", label_visibility="collapsed", type="jsonl", accept_multiple_files=True, key="upload_files")
                upload_submitted = st.form_submit_button("Check File(s) and Upload")
            if upload_submitted:
                if len(upload_files) == 0:
                    st.warning("No files selected.")
                else:
                    errors = 0
                    for file in upload_files:
                        check_res = check_finetune_dataset(file)
                        if check_res["format_errors"]:
                            error_msg = f"File `{file.name}` has the following errors:\n"
                            for error in check_res["format_errors"]:
                                error_msg += f"- {error}: {check_res['format_errors'][error]}\n"
                            st.error(error_msg)
                            errors += 1
                        else:
                            st.success(f"`{file.name}` is valid.")
                        if check_res["format_warnings"]:
                            warning_msg = f"However, it has the following warnings:\n"
                            for warning in check_res["format_warnings"]:
                                warning_msg += f"- {warning}: {check_res['format_warnings'][warning]}\n"
                            st.warning(warning_msg)
                    if errors == 0:
                        with st.spinner("Uploading files..."):
                            for file in stqdm(upload_files):
                                await upload_file(file)
                        upload_countdown = st.empty()
                        for i in range(5, 0, -1):
                            upload_countdown.success(f"All files uploaded successfully. Refreshing file list in {i} seconds...")
                            await asyncio.sleep(1)
                        st.session_state["NEED_REFRESH"] = True
                        st.experimental_rerun()
                    else:
                        st.error(f"{errors} file(s) had errors. Please fix them and try again.")
        
            st.caption("Manage Files")
            if len(files_res["data"]) == 0:
                st.warning("No files found. Upload some data samples to enable model fine-tuning. For more info on how to prepare datasets, see: https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset")
            else:
                files_df = pd.DataFrame(files_res["data"])
                files_df = files_df[["created_at", "id", "filename", "bytes", "purpose"]].set_index("created_at")
                files_df.index = pd.to_datetime(files_df.index, unit="s").rename("created at (UTC)")
                st.dataframe(files_df.sort_index(), use_container_width=True)

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
                st.session_state["file_id"] = file_id
                if len(file_id) == 0:
                    st.error("Please enter a valid File ID.")
                elif st.session_state["file_id"] not in files_df.id.tolist():
                    st.error(f"File ID `{st.session_state['file_id']}` not found.")
                    if "tokens_estimate" in st.session_state:
                        del st.session_state["tokens_estimate"]
                elif not is_num(epochs) or int(epochs) < 1 or int(epochs) > 25:
                    st.error(f"Invalid value {epochs} for `epochs` (1 <= `epochs` <= 25)")
                elif action == "review":
                    epochs = int(epochs)
                    contents_res = await view_file_contents(st.session_state["file_id"])
                    if contents_res is None:
                        st.error(f"File ID `{st.session_state['file_id']}` not found.")
                        if "tokens_estimate" in st.session_state:
                            del st.session_state["tokens_estimate"]
                    elif isinstance(contents_res, dict) and "error" in contents_res:
                        st.error(f"{contents_res['error']['message']}")
                    else:

                        # Parse data
                        try:
                            lines = contents_res.decode("utf-8").split("\n")
                            dataset = [json.loads(line) for line in lines]
                            data_type = "JSONL"
                        except:
                            # Not a valid JSONL file, try CSV instead
                            csv_df = pd.read_csv(StringIO(contents_res.decode("utf-8")), index_col=0)
                            data_type = "CSV"
    
                        st.write(f"`{st.session_state['file_id']}`")
                        if data_type == "CSV":
                            data_col, chart_col = st.columns(2)
                            with data_col:
                                st.dataframe(csv_df.sort_index(), use_container_width=True)
                            with chart_col:
                                loss_cols = [col for col in csv_df.columns if "loss" in col]
                                if len(loss_cols) > 0:
                                    st.caption("Loss Curves")
                                    st.line_chart(csv_df[loss_cols])
                                
                                accuracy_cols = [col for col in csv_df.columns if "accuracy" in col]
                                if len(accuracy_cols) > 0:
                                    st.caption("Accuracy Curves")
                                    st.line_chart(csv_df[accuracy_cols])
                        else:
                            # JSONL data type
                            st.caption("File Contents (expand below area to view each training sample and their messages)")
                            st.json(dataset, expanded=False)
                            check_res = check_finetune_dataset(dataset)
                            if check_res["format_errors"]:
                                error_msg = f"File ID `{st.session_state['file_id']}` has the following errors:\n"
                                for error in check_res["format_errors"]:
                                    error_msg += f"- {error}: {check_res['format_errors'][error]}\n"
                                st.error(error_msg)
                                if "tokens_estimate" in st.session_state:
                                    del st.session_state["tokens_estimate"]
                            else:
                                st.session_state["tokens_estimate"] = estimate_training_tokens(
                                    dataset = check_res["dataset"],
                                    convo_lens = check_res["convo_lens"],
                                    epochs=epochs
                                )
                            if check_res["format_warnings"]:
                                warning_msg = f"File ID `{st.session_state['file_id']}` has the following warnings:\n"
                                for warning in check_res["format_warnings"]:
                                    warning_msg += f"- {warning}: {check_res['format_warnings'][warning]}\n"
                                st.warning(warning_msg)
                elif action == "delete":
                    delete_res = await delete_file(file_id)
                    if isinstance(delete_res, dict) and "error" in delete_res:
                        st.error(f"{delete_res['error']['message']}")
                    elif delete_res is None:
                        st.error(f"File {file_id} deletion failed")
                    elif "deleted" in delete_res and not delete_res["deleted"]:
                        st.json(delete_res)
                    else:
                        if "tokens_estimate" in st.session_state:
                            del st.session_state["tokens_estimate"]
                        st.session_state["NEED_REFRESH"] = True
                        st.experimental_rerun()

            if "tokens_estimate" in st.session_state and len(st.session_state["tokens_estimate"]) > 0:
                tokens_estimate = st.session_state["tokens_estimate"]
                st.info(
                    f"Review passed. Dataset includes ~{tokens_estimate['n_billing_tokens']} tokens. By default, you'll train for {tokens_estimate['n_epochs']} epochs on this dataset. "
                    f"It amounts to ~{tokens_estimate['n_billing_tokens'] * tokens_estimate['n_epochs']} tokens in total. Check https://openai.com/pricing to estimate total costs.",
                    icon="✅"
                )
                with st.form("finetune-form", clear_on_submit=True):
                    st.caption("**Train a Finetuned Model**")
                    train_id_col, epochs_col2 = st.columns(2)
                    with train_id_col:
                        st.text_input("Training File ID (review another File ID to change)", value=st.session_state["file_id"], disabled=True)
                    with epochs_col2:
                        epochs = st.number_input(
                            "Actual Epochs to Train (1-25)",
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
                if finetune_submitted:
                    data = {
                        "training_file": st.session_state["file_id"],
                        "model": "gpt-3.5-turbo",
                        "hyperparameters": {
                            "n_epochs": epochs
                        }
                    }
                    if len(validation_id) > 0:
                        data["validation_file"] = validation_id
                    if len(suffix) > 0:
                        data["suffix"] = suffix
                    create_finetune_job_res = await create_finetune_job(data)
                    if isinstance(create_finetune_job_res, dict) and "error" in create_finetune_job_res:
                        st.error(f"{create_finetune_job_res['error']['message']}")
                    else:
                        finetune_countdown = st.empty()
                        for i in range(5, 0, -1):
                            finetune_countdown.success(f"Finetune job submission successful. Refreshing finetuning list in {i} seconds...")
                            await asyncio.sleep(1)
                        if "tokens_estimate" in st.session_state:
                            del st.session_state["tokens_estimate"]
                        st.session_state["NEED_REFRESH"] = True
                        st.experimental_rerun()

    with st.expander("**Model Playground**", expanded=True):
        models_col, chat_col = st.columns(2)
        with models_col:
            models_data = models_res["data"]
            # Each dict element in models_data has a "permission" key, which is a list of dicts,
            # we want to extract its key-value pairs to the top level of the dict
            for model in models_data:
                for i in range(len(model["permission"])):
                    for key, value in model["permission"][i].items():
                        model[f"perm_{i}_{key}"] = value
            models_df = pd.DataFrame(models_data)
            models_df["created"] = pd.to_datetime(models_df["created"], unit="s")
            models_df = models_df.set_index("created")
            models_df.index.rename("created at (UTC)", inplace=True)
            st.dataframe(models_df[["id", "owned_by", "root"]].sort_index(), use_container_width=True, height=300)

            with st.form("prompt_form", clear_on_submit=False):
                st.caption("**Test Model Performance**")
                model_id = st.selectbox("Model ID", models_df["id"].values, index=len(models_df) - 1)
                system_prompt = st.text_input("System Prompt (Optional)", help="You can use the System Prompt to guide model behavior by giving it some instructions.")
                prompt = st.text_area("Your Message")
                prompt_submitted = st.form_submit_button("Send")
        with chat_col:
            status_container = st.empty()
            chat_container = st.container()
            if "HISTORY" not in st.session_state:
                status_container.write("No messages yet, write something to start chatting.")
            else:
                with chat_container:
                    for message in st.session_state["HISTORY"]:
                        match message["role"]:
                            case "system":
                                status_container.write(f"**System Prompt:** {st.session_state['HISTORY'][0]['content']}")
                            case "user":
                                with st.chat_message("user"):
                                    st.markdown(message["content"])
                            case "assistant":
                                with st.chat_message("assistant", avatar="https://openai.com/favicon.ico"):
                                    st.markdown(message["content"])
            if st.button("Clear Messages"):
                if "HISTORY" in st.session_state:
                    del st.session_state["HISTORY"]
                st.experimental_rerun()
        if prompt_submitted:
            if len(prompt) == 0:
                models_col.error("Message cannot be empty.")
                st.stop()
            status_container.empty()
            if "HISTORY" not in st.session_state:
                st.session_state["HISTORY"] = []
                if len(system_prompt) > 0:
                    st.session_state["HISTORY"].append({
                        "role": "system",
                        "content": system_prompt
                    })
                st.session_state["HISTORY"].append({
                    "role": "user",
                    "content": prompt
                })
            else:
                if len(system_prompt) > 0:
                    if st.session_state["HISTORY"][0]["role"] != "system":
                        # There was no system prompt before but now we need to add it on the 0th index
                        st.session_state["HISTORY"].insert(0, {
                            "role": "system",
                            "content": system_prompt
                        })
                    else:
                        # Just replace the current system prompt
                        st.session_state["HISTORY"][0]["content"] = system_prompt
                elif len(system_prompt) == 0:
                    if st.session_state["HISTORY"][0]["role"] == "system":
                        # There was a system prompt before but now we need to remove it
                        del st.session_state["HISTORY"][0]
                st.session_state["HISTORY"].append({
                    "role": "user",
                    "content": prompt
                })
            if st.session_state["HISTORY"][0]["role"] == "system":
                status_container.write(f"**System Prompt:** {st.session_state['HISTORY'][0]['content']}")
            with chat_container:
                # Render the latest human message
                with st.chat_message("user"):
                    st.markdown(prompt)
                # Render the bot reply
                reply_box = st.empty()
                with reply_box:
                    with st.chat_message("assistant", avatar="https://openai.com/favicon.ico"):
                        loading_fn = FILE_ROOT / "loading.gif"
                        st.markdown(f"<img src='data:image/gif;base64,{get_local_img(loading_fn)}' width=30 height=10>", unsafe_allow_html=True)

                # Call the OpenAI API for final result
                reply_text = ""
                async for chunk in await openai.ChatCompletion.acreate(
                    model=model_id,
                    messages=st.session_state["HISTORY"],
                    stream=True,
                    timeout=TIMEOUT,
                ):
                    content = chunk["choices"][0].get("delta", {}).get("content", None)
                    if content is not None:
                        reply_text += content

                        # Continuously render the reply as it comes in
                        with reply_box:
                            with st.chat_message("assistant", avatar="https://openai.com/favicon.ico"):
                                st.markdown(reply_text)

                # Final fixing
                reply_text = reply_text.strip()

                with reply_box:
                    with st.chat_message("assistant", avatar="https://openai.com/favicon.ico"):
                        st.markdown(reply_text)

                # Append the final reply to the chat history
                st.session_state["HISTORY"].append({
                    "role": "assistant",
                    "content": reply_text
                })

                st.experimental_rerun()
            
asyncio.run(main())