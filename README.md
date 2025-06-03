# OpenAI Usage API Dashboard

A powerful Streamlit application for fetching, analyzing, and exporting usage data from OpenAI's comprehensive Usage API across all available endpoints.

## 🚀 What This App Does

This dashboard provides a unified interface to query OpenAI's Usage API endpoints and generate comprehensive usage reports. It fetches data from multiple usage endpoints simultaneously, enriches the data with human-readable names, and exports everything to CSV for further analysis.

### Supported Usage Endpoints

- **Completions** - Text generation usage (input/output tokens, cached tokens, audio tokens)
- **Embeddings** - Text embedding usage 
- **Moderations** - Content moderation usage
- **Images** - Image generation/editing usage
- **Audio Speeches** - Text-to-speech usage
- **Audio Transcriptions** - Speech-to-text usage  
- **Vector Stores** - Vector storage usage
- **Code Interpreter Sessions** - Code execution session usage

## ✨ Key Features

- **Async Processing**: Fetches data from multiple endpoints simultaneously with real-time progress tracking
- **Smart Caching**: Uses LRU cache to avoid redundant API calls and improve performance
- **Data Enrichment**: Automatically looks up and adds human-readable names for users, projects, and API keys
- **Flexible Date Ranges**: Select any date range for usage analysis
- **Endpoint Selection**: Choose which usage endpoints to query
- **Column Reordering**: Presents data in a logical, analysis-friendly column order
- **CSV Export**: Download complete usage data for further analysis
- **Session Persistence**: Maintains API client and cache across app refreshes

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8+
- OpenAI Admin API Key

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Environment Setup (Optional)

Set your OpenAI Admin API key as an environment variable:

```bash
export OPENAI_ADMIN_KEY="sk-admin-your-key-here"
```

Or create a `devenv.sh` file:
```bash
OPENAI_ADMIN_KEY="sk-admin-your-key-here"
source devenv.sh
```

## 🎯 How to Use

### 1. Start the Application

```bash
streamlit run app.py
```

### 2. Configure Your API Key

- If you set the `OPENAI_ADMIN_KEY` environment variable, it will be pre-populated
- Otherwise, enter your OpenAI Admin API key in the sidebar

### 3. Select Your Query Parameters

- **Date Range**: Choose start and end dates (treated as UTC)
- **Usage Endpoints**: Select which usage types to fetch (all selected by default)

### 4. Fetch Data

Click "Fetch Usage Data" to:
- Query selected endpoints asynchronously
- Fetch lookup data for users, projects, and API keys
- Process and enrich the data
- Display results with progress tracking

### 5. Analyze and Export

- Review the data in the interactive table
- Download as CSV for further analysis in Excel, Google Sheets, or other tools

## 📊 Data Output

The final dataset includes:

### Core Usage Data
- Date, endpoint type, token counts, model requests
- Input/output tokens, cached tokens, audio tokens
- Model names, batch status

### Enriched Information  
- **User Details**: User ID → Email address
- **Project Details**: Project ID → Project name
- **API Key Details**: API Key ID → API key name

### Column Order
Data is presented with the most important columns first: date, endpoint_type, input_tokens, output_tokens, api_key_id, api_key_name, project_id, project_name, and other relevant metrics.

## 🔧 Advanced Features

### Cache Management
- Use the "Clear All Cache" button in the sidebar to reset cached data
- Cache persists across app refreshes for improved performance
- View cache statistics when `LOGURU_LEVEL=DEBUG` is set

### Special Endpoint Handling
- Vector stores and code interpreter sessions are automatically configured with appropriate grouping parameters
- All other endpoints group by user_id, project_id, api_key_id, and model for maximum granularity

## 🚨 Requirements

- Valid OpenAI Admin API key with usage access
- Network access to OpenAI's API endpoints
- Python packages listed in `requirements.txt`

## 📝 Notes

- Dates are treated as UTC when converting to timestamps
- The app handles pagination automatically across all endpoints
- Empty results are handled gracefully with appropriate user feedback
- All errors are captured and displayed with detailed tracebacks

## 🤝 Usage Tips

- Run similar queries multiple times to benefit from caching
- Use the debug mode to monitor cache performance
- Select specific endpoints if you only need certain usage types
- Export data regularly for historical analysis and reporting

---

Built with ❤️ using Streamlit, httpx, and async processing for optimal performance.