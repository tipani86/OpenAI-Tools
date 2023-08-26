import os
import json
import openai
import aiohttp
import asyncio
import inspect
import argparse
import tiktoken
import pandas as pd
from stqdm import stqdm
from loguru import logger
from random import randint
from collections import defaultdict
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential

TIMEOUT = 60
N_RETRIES = 3
BACKOFF = 5
MULTIPLIER = 1.5

# Token counting functions
encoding = tiktoken.get_encoding("cl100k_base")

# not exact!
# simplified from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
def num_tokens_from_messages(messages, tokens_per_message=3, tokens_per_name=1):
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens

def num_assistant_tokens_from_messages(messages):
    num_tokens = 0
    for message in messages:
        if message["role"] == "assistant":
            num_tokens += len(encoding.encode(message["content"]))
    return num_tokens

def check_finetune_dataset(data) -> dict:
    # Adapted from https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset

    if not isinstance(data, list):
        dataset = [json.loads(line) for line in data]
    else:
        dataset = data

    # Format error checks and warnings and statistics
    format_errors = defaultdict(int)
    format_warnings = defaultdict(int)

    n_messages = []
    convo_lens = []
    assistant_message_lens = []

    for ex in dataset:
        if not isinstance(ex, dict):
            format_errors["data_type"] += 1
            continue

        messages = ex.get("messages", None)
        if not messages:
            format_errors["missing_messages_list"] += 1
            continue

        for message in messages:
            if "role" not in message or "content" not in message:
                format_errors["message_missing_key"] += 1

            if any(k not in ("role", "content", "name") for k in message):
                format_errors["message_unrecognized_key"] += 1

            if message.get("role", None) not in ("system", "user", "assistant"):
                format_errors["unrecognized_role"] += 1

            content = message.get("content", None)
            if not content or not isinstance(content, str):
                format_errors["missing_content"] += 1

        if not any(message.get("role", None) == "system" for message in messages):
            format_warnings["warn_n_missing_system"] += 1
        if not any(message.get("role", None) == "user" for message in messages):
            format_warnings["warn_n_missing_user"] += 1
        if not any(message.get("role", None) == "assistant" for message in messages):
            format_warnings["warn_n_missing_assistant_message"] += 1
        n_messages.append(len(messages))
        convo_lens.append(num_tokens_from_messages(messages))
        assistant_message_lens.append(num_assistant_tokens_from_messages(messages))

    n_too_long = sum(1 for l in convo_lens if l > 4096)
    if n_too_long > 0:
        format_errors["n_too_long"] = n_too_long

    if len(dataset) < 10:
        format_errors["less_than_10_examples"] = True

    res = {
        "n_messages": n_messages,
        "convo_lens": convo_lens,
        "assistant_message_lens": assistant_message_lens,
        "format_errors": format_errors,
        "format_warnings": format_warnings,
        "dataset": dataset,
    }

    return res

def estimate_training_tokens(
    dataset: list[dict],
    convo_lens: list[int],
    epochs: int=3
) -> dict:
    # Adapted from https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset

    # Pricing and default n_epochs estimate
    MAX_TOKENS_PER_EXAMPLE = 4096

    MIN_TARGET_EXAMPLES = 100
    MAX_TARGET_EXAMPLES = 25000
    TARGET_EPOCHS = epochs
    MIN_EPOCHS = 1
    MAX_EPOCHS = 25

    n_epochs = TARGET_EPOCHS
    n_train_examples = len(dataset)
    if n_train_examples * TARGET_EPOCHS < MIN_TARGET_EXAMPLES:
        n_epochs = min(MAX_EPOCHS, MIN_TARGET_EXAMPLES // n_train_examples)
    elif n_train_examples * TARGET_EPOCHS > MAX_TARGET_EXAMPLES:
        n_epochs = max(MIN_EPOCHS, MAX_TARGET_EXAMPLES // n_train_examples)

    n_billing_tokens = sum(min(MAX_TOKENS_PER_EXAMPLE, length) for length in convo_lens)

    res = {
        "n_billing_tokens": n_billing_tokens,
        "n_epochs": n_epochs,
    }

    return res

class OpenAITools:

    def __init__(self,
        openai_api_endpoint: str = "https://api.openai.com/v1"
    ):
        self.openai_api_endpoint = openai_api_endpoint

    async def get_users(self,
        openai_api_key: str,
        openai_org_id: str,
    ) -> dict:
        path = f"/organizations/{openai_org_id}/users"
        res = await self.request(openai_api_key, openai_org_id, "GET", path)
        # Users data are under the "members" object, plus we also need
        # to flatten the "user" object inside each member
        output = []
        users = res["members"]
        for user in users["data"]:
            for k, v in user["user"].items():
                user[k] = v
            del user["user"]
            output.append(user)
        users["data"] = output
        return users
    
    async def get_files(self,
        openai_api_key: str,
        openai_org_id: str,
    ) -> dict:
        path = "/files"
        headers = {
            "Authorization": f"Bearer {openai_api_key}",
        }
        return await self.request(
            openai_api_key, openai_org_id, "GET", path,
            headers=headers)
    
    async def view_file_contents(self,
        openai_api_key: str,
        openai_org_id: str,
        file_id: str,
    ):
        path = f"/files/{file_id}/content"
        headers = {
            "Authorization": f"Bearer {openai_api_key}",
        }
        return await self.request(
            openai_api_key, openai_org_id, "GET", path,
            headers=headers)

    async def delete_file(self,
        openai_api_key: str,
        openai_org_id: str,
        file_id: str,
    ):
        path = f"/files/{file_id}"
        headers = {
            "Authorization": f"Bearer {openai_api_key}",
        }
        return await self.request(
            openai_api_key, openai_org_id, "DELETE", path,
            headers=headers)
    
    async def upload_file(self,
        openai_api_key: str,
        openai_org_id: str,
        file,
    ):
        openai.api_key = openai_api_key
        openai.organization = openai_org_id
        await openai.File.acreate(
            file=file.getvalue(),
            purpose="fine-tune",
            user_provided_filename=file.name)
    
    async def get_finetune_jobs(self,
        openai_api_key: str,
        openai_org_id: str,
    ) -> dict:
        path = "/fine_tuning/jobs"
        headers = {
            "Authorization": f"Bearer {openai_api_key}",
        }
        return await self.request(
            openai_api_key, openai_org_id, "GET", path,
            headers=headers)
    
    async def create_finetune_job(self,
        openai_api_key: str,
        openai_org_id: str,
        data: dict,
    ):
        path = "/fine_tuning/jobs"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}",
        }
        return await self.request(
            openai_api_key, openai_org_id, "POST", path,
            headers=headers, data=data)
        
    async def cancel_finetune_job(self,
        openai_api_key: str,
        openai_org_id: str,
        job_id: str,
    ):
        path = f"/fine_tuning/jobs/{job_id}/cancel"
        headers = {
            "Authorization": f"Bearer {openai_api_key}",
        }
        return await self.request(
            openai_api_key, openai_org_id, "POST", path,
            headers=headers)
    
    async def get_models(self,
        openai_api_key: str,
        openai_org_id: str,
    ) -> dict:
        path = "/models"
        headers = {
            "Authorization": f"Bearer {openai_api_key}",
        }
        return await self.request(
            openai_api_key, openai_org_id, "GET", path,
            headers=headers)
    
    async def get_overall_usage(self,
        openai_api_key: str,
        openai_org_id: str,
        date_range: list[str],
    ):
        path = "/dashboard/billing/usage"
        params = {
            "start_date": date_range[0],
            "end_date": date_range[1],
        }
        res = await self.request(openai_api_key, openai_org_id, "GET", path, params=params)
        # Costs are under the "daily_costs" object, plus we also need extract each daily line item
        usage = {"object": "list", "data": []}
        for day in res["daily_costs"]:
            timestamp = day["timestamp"]
            for line_item in day["line_items"]:
                item = {"timestamp": timestamp}
                item.update(line_item)
                usage["data"].append(item)
        return usage
    
    async def get_usage(self,
        openai_api_key: str,
        openai_org_id: str,
        user_id: str,
        date_range: list[str],
    ):
        path = f"/usage"
        # Convert the date range items into start_date and end_date datetime objects and generate
        # also the intermediate dates, one per day, then export them as a list of strings (end date inclusive)
        start_date = datetime.strptime(date_range[0], "%Y-%m-%d")
        end_date = datetime.strptime(date_range[1], "%Y-%m-%d")
        dates = [(start_date + timedelta(days=x)).strftime("%Y-%m-%d") for x in range(0, (end_date-start_date).days+1)]
        # Iterate over the date range and get the usage for each day
        usage = {"object": "list", "data": []}
        logger.info(f"Getting usage for ID {user_id} in date range {date_range}")
        for date in stqdm(dates):
            params = {
                "date": date,
                "user_public_id": user_id,
            }
            res = await self.request(openai_api_key, openai_org_id, "GET", path, params=params)
            usage["data"].extend(res["data"])
            # The usage API rate limit is 5 rpm, so we need to space out the requests by at least 12 seconds
            await asyncio.sleep(randint(12, 15))
        return usage

    @retry(stop=stop_after_attempt(N_RETRIES), wait=wait_exponential(multiplier=MULTIPLIER, min=BACKOFF), reraise=True, retry_error_callback=logger.error)
    async def request(self,
        openai_api_key: str,
        openai_org_id: str,
        method: str,
        path: str,
        params: dict = {},
        headers: dict = {},
        data: aiohttp.FormData | dict | None = None,
    ) -> dict | bytes:
        headers.update({
            "authorization": f"Bearer {openai_api_key}",
            "openai-organization": openai_org_id,
            "authority": "api.openai.com",
            "method": method,
        })
        uri = f"{self.openai_api_endpoint}{path}"
        logger.debug(f"Requesting {method} {uri} with params {params} and headers {headers}")
        if data:
            logger.debug(f"Request data: {data}")
        
        async with aiohttp.ClientSession() as session:
            if isinstance(data, aiohttp.FormData):
                sess_func = session.request(method, uri, params=params, headers=headers, data=data)
            elif isinstance(data, dict):
                sess_func = session.request(method, uri, params=params, headers=headers, json=data)
            elif data is None:
                sess_func = session.request(method, uri, params=params, headers=headers)
            async with sess_func as resp:
                if resp.status == 200:
                    content_type = resp.headers["Content-Type"]
                    if content_type == "application/json":
                        return await resp.json()
                    else:
                        return await resp.read()
                elif resp.status == 400:
                    return await resp.json()
                else:
                    raise Exception(f"Request failed with status {resp.status}: {resp.reason}, {await resp.text()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", help="Action to perform", choices=[
        "get_overall_usage", "get_users", "get_usage",
    ])
    parser.add_argument("-e", "--endpoint", help="OpenAI API endpoint", default="https://api.openai.com/v1")
    parser.add_argument("-u", "--user_id", default=None, help="User ID to get usage for")
    parser.add_argument("-d", "--date_range", nargs=2, default=None, help="Date range to get usage for (in YYYY-MM-DD format)")
    args = parser.parse_args()

    errors = []
    for env_key in ["OPENAI_API_KEY", "OPENAI_ORGANIZATION"]:
        if not os.environ.get(env_key, None):
            errors.append(f"Environment variable {env_key} is not set")
    if errors:
        logger.error("Errors found:")
        for error in errors:
            logger.error(error)

    openai_tool_op = OpenAITools(args.endpoint.rstrip("/"))
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    openai_org_id = os.environ.get("OPENAI_ORGANIZATION")

    # Check the validity of date range format (YYYY-MM-DD) using strptime
    if args.date_range:
        for date in args.date_range:
            try:
                datetime.strptime(date, "%Y-%m-%d")
            except ValueError:
                raise Exception(f"Date {date} is not in YYYY-MM-DD format")

    match args.action:
        case "get_users":
            res = asyncio.run(openai_tool_op.get_users(openai_api_key, openai_org_id))
        case "get_overall_usage":
            # if not args.user_id:
            #     raise Exception("User ID is required for get_usage")
            if not args.date_range:
                raise Exception("Date range is required for get_usage")
            res = asyncio.run(openai_tool_op.get_overall_usage(openai_api_key, openai_org_id, args.date_range))
        case "get_usage":
            if not args.user_id:
                raise Exception("User ID is required for get_usage")
            if not args.date_range:
                raise Exception("Date range is required for get_usage")
            res = asyncio.run(openai_tool_op.get_usage(openai_api_key, openai_org_id, args.user_id, args.date_range))         

    logger.debug(f"Result: {res}")

    # Read the result data as a pandas dataframe and export it as a CSV file

    df = pd.DataFrame(res["data"])
    print(df)
    print(df.describe())

    df.to_csv(f"{args.action}.csv", index=False)