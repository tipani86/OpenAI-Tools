import os
import aiohttp
import asyncio
import argparse
import pandas as pd
from stqdm import stqdm
from loguru import logger
from random import randint
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential

TIMEOUT = 60
N_RETRIES = 1
BACKOFF = 5
MULTIPLIER = 1.5

class OpenAITools:

    def __init__(self,
        openai_api_endpoint: str = "https://api.openai.com/v1"
    ):
        self.openai_api_endpoint = openai_api_endpoint

    async def get_users(self,
        openai_api_key: str,
        openai_org_id: str,
    ):
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
    
    async def get_overall_usage(self,
        openai_api_key: str,
        openai_org_id: str,
        date_range: list[str],
    ):
        path = f"/dashboard/billing/usage"
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
        data: dict | None = None,
    ):
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
            async with session.request(method, uri, params=params, headers=headers, json=data) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    raise Exception(f"Request failed with status {resp.status}")

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