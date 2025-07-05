# 🚀 Streamlit Cloud Deployment Guide

## Prerequisites

1. **GitHub Repository**: Your code must be in a public GitHub repository
2. **Streamlit Account**: Sign up at [share.streamlit.io](https://share.streamlit.io)
3. **Environment Variables**: Set up your API keys in Streamlit Cloud

## Step-by-Step Deployment

### 1. Prepare Your Repository

Ensure your repository has these files:
- ✅ `app.py` (main Streamlit app)
- ✅ `requirements.txt` (Python dependencies)
- ✅ `.streamlit/config.toml` (Streamlit configuration)
- ✅ `packages.txt` (system dependencies, if needed)

### 2. Deploy on Streamlit Cloud

1. **Go to Streamlit Cloud**: Visit [share.streamlit.io](https://share.streamlit.io)
2. **Sign in**: Use your GitHub account
3. **New App**: Click "New app"
4. **Repository**: Select your Facebook-Social-Analyzer repository
5. **Branch**: Select `main`
6. **Main file path**: Enter `app.py`
7. **Advanced settings**: Click "Advanced settings"

### 3. Configure Environment Variables

In the advanced settings, add these secrets:

```toml
FACEBOOK_ACCESS_TOKEN = "your_facebook_access_token"
GEMINI_API_KEY = "your_gemini_api_key"
DATABASE_URL = "sqlite:///facebook_ads.db"
```

### 4. Deploy

Click "Deploy!" and wait for the build to complete.

## Environment Variables Setup

### Facebook Access Token
1. Go to [Facebook Developers](https://developers.facebook.com/)
2. Create/select your app
3. Add Marketing API product
4. Generate access token with permissions:
   - `ads_read`
   - `ads_management`
   - `business_management`

### Google Gemini API Key
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the key to Streamlit secrets

## Deployment Troubleshooting

### Common Issues

1. **Build Failures**
   - Check `requirements.txt` for correct dependencies
   - Ensure all imports are available
   - Verify Python version compatibility

2. **Environment Variables**
   - Double-check secret names match your code
   - Ensure tokens are valid and have proper permissions
   - Test locally with `.env` file first

3. **Database Issues**
   - SQLite works well on Streamlit Cloud
   - For PostgreSQL, use external database service
   - Ensure database URL is correct

### Performance Optimization

1. **Caching**: Use `@st.cache_data` for expensive operations
2. **Lazy Loading**: Load data only when needed
3. **Error Handling**: Graceful fallbacks for API failures

## Post-Deployment

### Testing Your App
1. **Basic Functionality**: Test data fetching and display
2. **AI Features**: Verify Gemini integration works
3. **Error Handling**: Test with invalid tokens/data
4. **Performance**: Check loading times and responsiveness

### Monitoring
- **Streamlit Cloud Dashboard**: Monitor app performance
- **Logs**: Check for errors and warnings
- **Usage**: Track user interactions

## Security Best Practices

1. **Never commit secrets**: Keep API keys in Streamlit secrets
2. **Validate inputs**: Sanitize user inputs
3. **Rate limiting**: Implement API call limits
4. **Error messages**: Don't expose sensitive information

## Custom Domain (Optional)

1. **Domain Setup**: Configure custom domain in Streamlit Cloud
2. **SSL Certificate**: Automatic HTTPS with custom domains
3. **DNS Configuration**: Point domain to Streamlit Cloud

## Support

- **Streamlit Documentation**: [docs.streamlit.io](https://docs.streamlit.io)
- **Community Forum**: [discuss.streamlit.io](https://discuss.streamlit.io)
- **GitHub Issues**: Report bugs in your repository

---

**Your Facebook Ads Analytics platform will be live at:**
`https://your-app-name.streamlit.app` 