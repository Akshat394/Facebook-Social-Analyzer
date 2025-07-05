# Facebook Social Analyzer

An advanced Facebook Ads Analytics platform with AI-powered features for comprehensive marketing intelligence.

## 🚀 Features

- **AI-Powered Analytics**: Natural language querying with Google Gemini
- **Real-time Data Fetching**: Direct integration with Facebook Marketing API
- **Advanced Visualizations**: Interactive charts and dashboards
- **Predictive Analytics**: Performance forecasting and trend analysis
- **Anomaly Detection**: Identify unusual patterns in your ad performance
- **Clustering Analysis**: Group similar campaigns for better optimization
- **ROI Optimization**: AI-powered recommendations for better returns
- **Seasonal Pattern Detection**: Optimize for seasonal trends

## 📋 Prerequisites

- Python 3.8 or higher
- Facebook Ads Account with API access
- Google Gemini API key

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

### Google Gemini API Setup
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Add the key to your `.env` file

## 🚀 Usage

### Standard Dashboard
```bash
streamlit run app.py
```

### Advanced Dashboard
```bash
streamlit run advanced_dashboard.py
```

### API Server
```bash
python api.py
```

## 📊 Dashboard Features

### Standard Dashboard
- Campaign overview and metrics
- AI-powered natural language queries
- Interactive charts and visualizations
- Raw data exploration

### Advanced Dashboard
- Predictive analytics and forecasting
- Anomaly detection and alerts
- Clustering analysis
- ROI optimization recommendations
- Seasonal pattern analysis
- Advanced AI insights

## 🔍 API Endpoints

The platform includes a REST API for programmatic access:

- `GET /api/campaigns` - Get all campaigns
- `GET /api/adsets` - Get all ad sets
- `GET /api/ads` - Get all ads
- `POST /api/query` - AI-powered query endpoint

## 🏗️ Architecture

```
├── app.py                 # Standard dashboard
├── advanced_dashboard.py  # Advanced AI dashboard
├── api.py                # REST API server
├── database.py           # Database management
├── facebook_api.py       # Facebook API integration
├── gemini_query.py       # AI query engine
├── utils.py              # Utility functions
└── requirements.txt      # Dependencies
```

## 🤖 AI Capabilities

- **Natural Language Processing**: Ask questions in plain English
- **Predictive Analytics**: Forecast future performance
- **Anomaly Detection**: Identify unusual patterns
- **Clustering Analysis**: Group similar campaigns
- **Trend Forecasting**: Predict market trends
- **ROI Optimization**: Maximize return on ad spend
- **Seasonal Analysis**: Detect seasonal patterns

## 📈 Key Metrics

- Campaign performance analysis
- Cost per acquisition (CPA)
- Return on ad spend (ROAS)
- Click-through rate (CTR)
- Conversion rate optimization
- Budget allocation insights

## 🔒 Security

- Environment variable configuration
- Secure API token handling
- Database connection security
- Input validation and sanitization

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

**Built with ❤️ using Streamlit, Plotly, and Google Gemini AI**