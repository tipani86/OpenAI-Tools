import httpx
from async_lru import alru_cache
import asyncio
import json
from loguru import logger
from typing import Dict, List, Any


class OpenAIUsageAPIClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            timeout=30.0
        )
    
    async def _fetch_page(self, endpoint: str, params: dict) -> dict:
        """Single page fetch with error handling"""
        response = await self.client.get(endpoint, params=params)
        response.raise_for_status()
        return response.json()
    
    @alru_cache(maxsize=128)
    async def fetch_all_pages(self, endpoint_name: str, endpoint_url: str, params_json: str) -> List[Dict[str, Any]]:
        """Paginated fetch helper that loops through all pages"""
        logger.debug(f"Fetching all pages for {endpoint_name} with params: {params_json}")
        # Convert JSON string back to dict for processing
        params = json.loads(params_json)
        
        all_results = []
        current_params = params.copy()
        
        while True:
            response_data = await self._fetch_page(endpoint_url, current_params)
            
            # Process each bucket in the response
            for bucket in response_data.get("data", []):
                start_time = bucket.get("start_time")
                results = bucket.get("results", [])
                
                # Each result object becomes a row with inherited start_time
                for result in results:
                    row = result.copy()
                    row["start_time"] = start_time
                    row["endpoint_type"] = endpoint_name
                    all_results.append(row)
            
            # Check for pagination
            if not response_data.get("has_more", False):
                break
                
            next_page = response_data.get("next_page")
            if not next_page:
                break
                
            current_params["page"] = next_page
        
        return all_results
    
    @alru_cache(maxsize=32)
    async def fetch_lookup_data(self, lookup_type: str, endpoint_url: str) -> Dict[str, str]:
        """Fetch lookup data for users, projects, or API keys"""
        all_items = []
        params = {"limit": 100}
        
        while True:
            response_data = await self._fetch_page(endpoint_url, params)
            all_items.extend(response_data.get("data", []))
            
            if not response_data.get("has_more", False):
                break
                
            # Use the last_id for pagination (different from usage endpoints)
            last_id = response_data.get("last_id")
            if not last_id:
                break
            params["after"] = last_id
        
        # Create ID to name mapping based on lookup type
        lookup_map = {}
        for item in all_items:
            if lookup_type == "users":
                lookup_map[item.get("id")] = item.get("email", "Unknown")
            elif lookup_type == "projects":
                lookup_map[item.get("id")] = item.get("name", "Unknown")
            elif lookup_type == "api_keys":
                lookup_map[item.get("id")] = item.get("name", "Unknown")
        
        return lookup_map
    
    async def close(self):
        """Cleanup method for httpx client"""
        await self.client.aclose() 