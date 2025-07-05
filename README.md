# Facebook Social Analyzer

An advanced Facebook Ads Analytics platform with AI-powered features for comprehensive marketing intelligence and robust data analysis.

## 🚀 Features

- **🤖 AI-Powered Analytics**: Natural language querying with Google Gemini
- **📊 Robust Data Handling**: Dynamic column detection and fallback logic
- **🔄 Real-time Data Fetching**: Direct integration with Facebook Marketing API
- **📈 Advanced Visualizations**: Interactive charts and dashboards
- **🔍 Smart Query Engine**: Intelligent analysis of campaign structure and metadata
- **📱 User-Friendly Interface**: Intuitive Streamlit dashboard with error handling
- **💾 Database Storage**: SQLite/PostgreSQL support with automatic schema management
- **🎯 Performance Insights**: Campaign, ad set, and ad analysis without requiring insights data

## 📋 Prerequisites

- Python 3.8 or higher
- Facebook Ads Account with API access
- Google Gemini API key (optional, for AI features)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Akshat394/Facebook-Social-Analyzer.git
   cd Facebook-Social-Analyzer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the project root:
   ```env
   FACEBOOK_ACCESS_TOKEN=your_facebook_access_token
   GEMINI_API_KEY=your_gemini_api_key
   DATABASE_URL=sqlite:///facebook_ads.db
   ```

## 🔧 Configuration

### Facebook API Setup
1. Go to [Facebook Developers](https://developers.facebook.com/)
2. Create a new app or use an existing one
3. Add the Marketing API product
4. Generate an access token with the following permissions:
   - `ads_read`
   - `ads_management`
   - `business_management`

### Google Gemini API Setup (Optional)
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Add the key to your `.env` file
4. AI features will work without this key, but with limited functionality

## 🚀 Usage

### Main Dashboard
```bash
streamlit run app.py
```

The dashboard will be available at `http://localhost:8501`

## 📊 Dashboard Features

### 🎯 **Overview Tab**
- **Dynamic Metrics**: Automatically detects available columns and displays relevant metrics
- **Smart Charts**: Visualizations that adapt to your data structure
- **Column Detection**: Works with any column naming convention from Facebook API
- **Fallback Logic**: Graceful handling when expected data is missing

### 🧠 **AI Query Tab**
- **Natural Language Queries**: Ask questions in plain English
- **Smart Analysis**: Understands your data structure and provides relevant insights
- **Example Queries**: Pre-built examples for common analysis tasks
- **Data Preview**: See available data for context
- **Error Handling**: Clear feedback when AI features aren't available

### 📋 **Raw Data Tab**
- **Complete Data View**: Explore all fetched data in organized tables
- **Campaigns Data**: Campaign metadata, status, objectives, budgets
- **Ad Sets Data**: Targeting, optimization goals, configurations
- **Ads Data**: Ad status, campaign relationships, creatives

## 🔍 AI Query Examples

### **Campaign Analysis**
- "How many campaigns do I have?"
- "Show me active campaigns"
- "Campaign status overview"
- "Campaign objectives breakdown"

### **Ad Analysis**
- "How many ads do I have?"
- "Show me active ads"
- "Ad distribution by campaign"
- "Ad status breakdown"

### **Budget & Performance**
- "Budget analysis"
- "Campaign performance analysis"
- "Recent campaigns and ads"
- "Active vs paused campaigns"

### **Ad Set Analysis**
- "Ad set optimization goals"
- "Show me active ad sets"
- "Targeting overview"

## 🏗️ Architecture

```
├── app.py                 # Main Streamlit dashboard
├── database.py           # Database management with SQLite/PostgreSQL
├── facebook_api.py       # Facebook Marketing API integration
├── gemini_query.py       # Advanced AI query engine
├── utils.py              # Utility functions and helpers
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🤖 AI Capabilities

### **Smart Query Engine**
- **Dynamic Schema Understanding**: Automatically adapts to your data structure
- **Fallback Queries**: Works even when insights data is unavailable
- **Natural Language Processing**: Understands complex queries in plain English
- **Contextual Analysis**: Provides insights based on available data

### **Data Analysis Features**
- **Campaign Structure Analysis**: Comprehensive overview of campaign hierarchy
- **Status Distribution**: Active vs paused analysis across campaigns, ads, and ad sets
- **Budget Analysis**: Campaign budget insights and allocation
- **Objective Analysis**: Campaign objective distribution and trends
- **Recent Activity**: Latest campaign and ad creation analysis

## 📈 Key Metrics & Insights

### **Available Without Performance Data**
- Campaign counts and status distribution
- Ad set optimization goals and targeting
- Ad distribution across campaigns
- Budget allocation and spending patterns
- Campaign objective analysis
- Recent activity and trends

### **Enhanced with Performance Data** (when available)
- Cost per acquisition (CPA)
- Return on ad spend (ROAS)
- Click-through rate (CTR)
- Conversion rate optimization
- Performance-based recommendations

## 🔒 Security & Reliability

- **Environment Variable Configuration**: Secure API token handling
- **Database Connection Security**: Protected database access
- **Input Validation**: Sanitized user inputs
- **Error Handling**: Graceful degradation when services are unavailable
- **Data Privacy**: Local data storage with optional cloud deployment

## 🚀 Recent Improvements

### **v2.0 - Enhanced Data Handling**
- ✅ **Robust Column Detection**: Works with any Facebook API column naming
- ✅ **Dynamic Dashboard**: Adapts to available data structure
- ✅ **Enhanced AI Queries**: Smart analysis of campaign metadata
- ✅ **Better Error Handling**: Clear feedback and graceful fallbacks
- ✅ **Improved User Experience**: Intuitive interface with helpful guidance

### **Key Features Added**
- Dynamic column detection and fallback logic
- Enhanced AI query engine with schema understanding
- Better error handling and user feedback
- Improved data visualization and analysis
- Comprehensive campaign structure analysis

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue on GitHub
- Check the documentation
- Review the troubleshooting guide

## 🔄 Updates

Stay updated with the latest features and improvements by:
- Starring the repository
- Watching for updates
- Following the release notes

---

**Built with ❤️ using Streamlit, Plotly, Google Gemini AI, and robust data handling**