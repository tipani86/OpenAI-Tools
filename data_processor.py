import pandas as pd
from datetime import datetime, timezone
from typing import List, Dict, Any


def unix_to_date_string(unix_timestamp: int) -> str:
    """Convert Unix timestamp to YYYY-MM-DD format"""
    return datetime.fromtimestamp(unix_timestamp, tz=timezone.utc).strftime("%Y-%m-%d")


def flatten_usage_data(raw_data: list, endpoint_name: str) -> list:
    """Flatten API response into list of dicts"""
    # This function is now handled in the API client, but keeping for consistency
    return raw_data


def enrich_dataframe_with_lookups(df: pd.DataFrame, user_lookup: Dict[str, str], 
                                project_lookup: Dict[str, str], api_key_lookup: Dict[str, str]) -> pd.DataFrame:
    """Add human-readable names to the dataframe based on ID lookups"""
    if df.empty:
        return df
    
    # Add user name column
    if "user_id" in df.columns:
        df["user_email"] = df["user_id"].map(user_lookup).fillna("Unknown")
    
    # Add project name column
    if "project_id" in df.columns:
        df["project_name"] = df["project_id"].map(project_lookup).fillna("Unknown")
    
    # Add API key name column
    if "api_key_id" in df.columns:
        df["api_key_name"] = df["api_key_id"].map(api_key_lookup).fillna("Unknown")
    
    return df


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder columns with specified columns first, followed by remaining columns"""
    if df.empty:
        return df
    
    # Desired column order (leftmost columns)
    preferred_order = [
        "date", "endpoint_type", "input_tokens", "output_tokens", 
        "api_key_id", "api_key_name", "project_id", "project_name", 
        "input_cached_tokens", "input_audio_tokens", "output_audio_tokens", 
        "model", "start_time", "user_email", "num_model_requests"
    ]

    # Banned columns
    banned_columns = [
        "object", "user_id", "batch"
    ]
    
    # Get columns that exist in the dataframe from the preferred order
    existing_preferred = [col for col in preferred_order if col in df.columns]
    
    # Get remaining columns not in the preferred order (those which are not banned)
    remaining_columns = [col for col in df.columns if (col not in preferred_order and col not in banned_columns)]
    
    # Combine preferred columns (leftmost) with remaining columns (rightmost)
    final_order = existing_preferred + remaining_columns
    
    return df[final_order]


def create_dataframe(all_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert list to DataFrame and sort by time"""
    if not all_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_data)
    
    # Convert start_time to date string format (in UTC time, ignore local timezone)
    if "start_time" in df.columns:
        df["date"] = df["start_time"].apply(unix_to_date_string)
        df = df.sort_values("start_time")
    
    return df 