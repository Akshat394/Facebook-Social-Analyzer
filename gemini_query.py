import os
from dotenv import load_dotenv
load_dotenv()
import google.generativeai as genai
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple
import re
from datetime import datetime, timedelta
import json
import pickle
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiQueryEngine:
    """Simple wrapper for the Advanced Gemini Query Engine"""
    
    def __init__(self, database_manager):
        self.advanced_engine = AdvancedGeminiQueryEngine(database_manager)
    
    def query(self, user_query: str) -> str:
        """Simple query method that returns a formatted response"""
        try:
            # Use the advanced engine to process the query
            result = self.advanced_engine.process_query(user_query)
            
            if result.get('error'):
                return f"❌ Error: {result['error']}"
            
            # Format the response
            response = []
            
            # Add insights
            if result.get('insights'):
                response.append("## 📊 Analysis Results")
                response.append(result['insights'])
            
            # Add recommendations
            if result.get('recommendations'):
                response.append("\n## 💡 Recommendations")
                for rec in result['recommendations'][:3]:  # Show top 3
                    response.append(f"- **{rec['title']}**: {rec['description']}")
            
            # Add data summary
            if result.get('data') is not None and not result['data'].empty:
                response.append(f"\n## 📋 Data Summary")
                response.append(f"Found {len(result['data'])} records matching your query.")
                
                # Show sample data
                if len(result['data']) > 0:
                    response.append("\n**Sample data:**")
                    sample_data = result['data'].head(5).to_string()
                    response.append(f"```\n{sample_data}\n```")
            
            # If no structured response, try to provide a basic analysis
            if not response:
                if result.get('data') is not None:
                    if not result['data'].empty:
                        response.append("## 📊 Query Results")
                        response.append(f"Found {len(result['data'])} records.")
                        response.append("\n**Data Preview:**")
                        response.append(f"```\n{result['data'].head(10).to_string()}\n```")
                    else:
                        response.append("## 📊 Query Results")
                        response.append("No data found matching your query.")
                else:
                    response.append("## 📊 Query Results")
                    response.append("Unable to process your query. Please try a different question.")
            
            return "\n".join(response) if response else "No results found for your query."
            
        except Exception as e:
            logger.error(f"Error in simple query: {e}")
            return f"❌ Error processing query: {str(e)}"

class AdvancedGeminiQueryEngine:
    def __init__(self, database_manager):
        self.db_manager = database_manager
        self.api_key = os.getenv('GEMINI_API_KEY', '')
        
        # Initialize multiple Gemini models for different tasks
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.text_model = genai.GenerativeModel('gemini-1.5-flash')
            self.pro_model = genai.GenerativeModel('gemini-1.5-pro')
            self.vision_model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            logger.warning("Gemini API key not found")

        self.schema_info = self.db_manager.get_schema_info()
        self._initialize_query_patterns()
        self._initialize_advanced_features()
        
        # Learning and memory system
        self.query_history = []
        self.user_preferences = {}
        self.performance_cache = {}
        
        # Load learned patterns
        self._load_learned_patterns()

    def _initialize_advanced_features(self):
        """Initialize advanced AI features"""
        self.advanced_features = {
            'predictive_analytics': True,
            'anomaly_detection': True,
            'clustering_analysis': True,
            'trend_forecasting': True,
            'competitive_analysis': True,
            'roi_optimization': True,
            'audience_insights': True,
            'creative_performance': True,
            'cross_channel_analysis': True,
            'seasonal_patterns': True
        }
        
        # Initialize ML models
        self.ml_models = {
            'anomaly_detector': IsolationForest(contamination=0.1, random_state=42),
            'performance_predictor': RandomForestRegressor(n_estimators=100, random_state=42),
            'clustering_model': KMeans(n_clusters=5, random_state=42),
            'scaler': StandardScaler()
        }

    def _load_learned_patterns(self):
        """Load learned query patterns and user preferences"""
        try:
            if os.path.exists('learned_patterns.pkl'):
                with open('learned_patterns.pkl', 'rb') as f:
                    data = pickle.load(f)
                    self.query_history = data.get('query_history', [])
                    self.user_preferences = data.get('user_preferences', {})
                    self.performance_cache = data.get('performance_cache', {})
        except Exception as e:
            logger.warning(f"Could not load learned patterns: {e}")

    def _save_learned_patterns(self):
        """Save learned patterns for future use"""
        try:
            data = {
                'query_history': self.query_history[-1000:],  # Keep last 1000 queries
                'user_preferences': self.user_preferences,
                'performance_cache': self.performance_cache
            }
            with open('learned_patterns.pkl', 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"Could not save learned patterns: {e}")

    def process_advanced_query(self, user_query: str, user_id: str = None) -> Dict[str, Any]:
        """Process query with advanced AI capabilities"""
        try:
            # Record query for learning
            self.query_history.append({
                'query': user_query,
                'user_id': user_id,
                'timestamp': datetime.now(),
                'context': self._extract_query_context(user_query)
            })
            
            # Enhanced query processing
            enhanced_query = self._enhance_query_with_ai(user_query)
            sql_query = self._generate_advanced_sql_query(enhanced_query)
            
            if not sql_query:
                return self._create_error_response("Failed to generate SQL query")

            # Execute query and get data
            data = self.db_manager.execute_query(sql_query)
            
            # Advanced analysis pipeline
            analysis_results = self._perform_advanced_analysis(data, user_query)
            
            # Generate comprehensive insights
            insights = self._generate_advanced_insights(user_query, data, sql_query, analysis_results)
            
            # Create visualizations
            visualizations = self._create_advanced_visualizations(data, analysis_results)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(data, analysis_results, user_query)
            
            # Update learning system
            self._update_learning_system(user_query, analysis_results)
            
            return {
                'sql_query': sql_query,
                'data': data,
                'insights': insights,
                'visualizations': visualizations,
                'recommendations': recommendations,
                'analysis_results': analysis_results,
                'enhanced_query': enhanced_query,
                'error': None
            }

        except Exception as e:
            logger.error(f"Error in advanced query processing: {e}")
            return self._create_error_response(str(e))

    def _enhance_query_with_ai(self, user_query: str) -> str:
        """Use AI to enhance and understand user queries"""
        try:
            # Basic enhancement
            enhanced = self._enhance_query(user_query)
            
            # AI-powered query understanding
            if hasattr(self, 'text_model'):
                understanding_prompt = f"""
                Analyze this Facebook Ads query and enhance it with business context:
                Original: {user_query}
                Enhanced: {enhanced}
                
                Provide additional context for:
                1. Business intent (what the user really wants to know)
                2. Related metrics that would be valuable
                3. Time context if not specified
                4. Comparative analysis opportunities
                5. Predictive insights that could be useful
                
                Return only the enhanced query with context.
                """
                
                response = self.text_model.generate_content(understanding_prompt)
                ai_enhanced = response.text.strip()
                
                if ai_enhanced and len(ai_enhanced) > len(enhanced):
                    return ai_enhanced
            
            return enhanced
            
        except Exception as e:
            logger.warning(f"AI enhancement failed: {e}")
            return self._enhance_query(user_query)

    def _perform_advanced_analysis(self, data: pd.DataFrame, user_query: str) -> Dict[str, Any]:
        """Perform comprehensive advanced analysis"""
        if data is None or data.empty:
            return {}
        
        results = {}
        
        try:
            # Anomaly Detection
            if self.advanced_features['anomaly_detection']:
                results['anomalies'] = self._detect_anomalies(data)
            
            # Predictive Analytics
            if self.advanced_features['predictive_analytics']:
                results['predictions'] = self._generate_predictions(data)
            
            # Clustering Analysis
            if self.advanced_features['clustering_analysis']:
                results['clusters'] = self._perform_clustering(data)
            
            # Trend Analysis
            if self.advanced_features['trend_forecasting']:
                results['trends'] = self._analyze_trends(data)
            
            # Performance Scoring
            results['performance_scores'] = self._calculate_performance_scores(data)
            
            # ROI Analysis
            if self.advanced_features['roi_optimization']:
                results['roi_analysis'] = self._analyze_roi(data)
            
            # Seasonal Patterns
            if self.advanced_features['seasonal_patterns']:
                results['seasonal_patterns'] = self._detect_seasonal_patterns(data)
            
        except Exception as e:
            logger.error(f"Advanced analysis error: {e}")
        
        return results

    def _detect_anomalies(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalies in the data"""
        try:
            # Select numeric columns for anomaly detection
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return {}
            
            # Prepare data for anomaly detection
            X = data[numeric_cols].fillna(0)
            X_scaled = self.ml_models['scaler'].fit_transform(X)
            
            # Detect anomalies
            anomalies = self.ml_models['anomaly_detector'].fit_predict(X_scaled)
            anomaly_indices = np.where(anomalies == -1)[0]
            
            return {
                'anomaly_count': len(anomaly_indices),
                'anomaly_percentage': len(anomaly_indices) / len(data) * 100,
                'anomaly_indices': anomaly_indices.tolist(),
                'anomaly_data': data.iloc[anomaly_indices].to_dict('records') if len(anomaly_indices) > 0 else []
            }
        except Exception as e:
            logger.error(f"Anomaly detection error: {e}")
            return {}

    def _generate_predictions(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate predictions for key metrics"""
        try:
            if 'spend' in data.columns and 'impressions' in data.columns:
                # Simple trend-based prediction
                recent_data = data.tail(7)  # Last 7 days
                if len(recent_data) >= 3:
                    spend_trend = np.polyfit(range(len(recent_data)), recent_data['spend'], 1)
                    impressions_trend = np.polyfit(range(len(recent_data)), recent_data['impressions'], 1)
                    
                    # Predict next 7 days
                    future_days = np.array(range(len(recent_data), len(recent_data) + 7))
                    predicted_spend = np.polyval(spend_trend, future_days)
                    predicted_impressions = np.polyval(impressions_trend, future_days)
                    
                    return {
                        'spend_prediction': predicted_spend.tolist(),
                        'impressions_prediction': predicted_impressions.tolist(),
                        'trend_direction': 'increasing' if spend_trend[0] > 0 else 'decreasing',
                        'confidence': min(abs(spend_trend[0]) / np.mean(recent_data['spend']) * 100, 95)
                    }
        except Exception as e:
            logger.error(f"Prediction error: {e}")
        return {}

    def _perform_clustering(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform clustering analysis on campaigns/ads"""
        try:
            # Select features for clustering
            features = ['spend', 'impressions', 'clicks', 'ctr', 'cpc']
            available_features = [f for f in features if f in data.columns]
            
            if len(available_features) < 2:
                return {}
            
            X = data[available_features].fillna(0)
            X_scaled = self.ml_models['scaler'].fit_transform(X)
            
            # Perform clustering
            clusters = self.ml_models['clustering_model'].fit_predict(X_scaled)
            data['cluster'] = clusters
            
            # Analyze clusters
            cluster_analysis = {}
            for cluster_id in range(self.ml_models['clustering_model'].n_clusters):
                cluster_data = data[data['cluster'] == cluster_id]
                cluster_analysis[f'cluster_{cluster_id}'] = {
                    'size': len(cluster_data),
                    'avg_spend': cluster_data['spend'].mean() if 'spend' in cluster_data.columns else 0,
                    'avg_ctr': cluster_data['ctr'].mean() if 'ctr' in cluster_data.columns else 0,
                    'characteristics': self._describe_cluster_characteristics(cluster_data)
                }
            
            return {
                'cluster_count': self.ml_models['clustering_model'].n_clusters,
                'cluster_analysis': cluster_analysis,
                'clustered_data': data.to_dict('records')
            }
        except Exception as e:
            logger.error(f"Clustering error: {e}")
            return {}

    def _analyze_trends(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trends in the data"""
        try:
            trends = {}
            
            # Time-based trend analysis
            if 'date_start' in data.columns:
                data['date_start'] = pd.to_datetime(data['date_start'])
                data = data.sort_values('date_start')
                
                # Daily trends
                daily_trends = data.groupby('date_start').agg({
                    'spend': 'sum',
                    'impressions': 'sum',
                    'clicks': 'sum'
                }).reset_index()
                
                # Calculate trend slopes
                for col in ['spend', 'impressions', 'clicks']:
                    if col in daily_trends.columns and len(daily_trends) > 1:
                        x = np.arange(len(daily_trends))
                        y = daily_trends[col].values
                        slope = np.polyfit(x, y, 1)[0]
                        trends[f'{col}_trend'] = {
                            'slope': slope,
                            'direction': 'increasing' if slope > 0 else 'decreasing',
                            'strength': abs(slope) / np.mean(y) if np.mean(y) > 0 else 0
                        }
            
            return trends
        except Exception as e:
            logger.error(f"Trend analysis error: {e}")
            return {}

    def _calculate_performance_scores(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate performance scores for campaigns/ads"""
        try:
            scores = {}
            
            # Overall performance score
            if 'ctr' in data.columns and 'cpc' in data.columns:
                # Normalize metrics (0-100 scale)
                ctr_score = np.clip(data['ctr'] * 100, 0, 100)
                cpc_score = np.clip((1 / data['cpc']) * 100, 0, 100)  # Inverse relationship
                
                # Weighted performance score
                performance_score = (ctr_score * 0.6 + cpc_score * 0.4).mean()
                scores['overall_performance'] = performance_score
            
            # Efficiency score
            if 'spend' in data.columns and 'impressions' in data.columns:
                efficiency = data['impressions'] / data['spend']
                scores['efficiency_score'] = efficiency.mean()
            
            # ROI score (if conversion data available)
            if 'conversion_values' in data.columns:
                roi = data['conversion_values'] / data['spend']
                scores['roi_score'] = roi.mean()
            
            return scores
        except Exception as e:
            logger.error(f"Performance scoring error: {e}")
            return {}

    def _analyze_roi(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze ROI and optimization opportunities"""
        try:
            roi_analysis = {}
            
            if 'spend' in data.columns and 'conversion_values' in data.columns:
                # Calculate ROI
                data['roi'] = data['conversion_values'] / data['spend']
                
                # ROI distribution
                roi_analysis['avg_roi'] = data['roi'].mean()
                roi_analysis['roi_distribution'] = {
                    'high_roi': len(data[data['roi'] > 3]),
                    'medium_roi': len(data[(data['roi'] >= 1) & (data['roi'] <= 3)]),
                    'low_roi': len(data[data['roi'] < 1])
                }
                
                # Optimization recommendations
                low_roi_high_spend = data[(data['roi'] < 1) & (data['spend'] > data['spend'].quantile(0.75))]
                roi_analysis['optimization_opportunities'] = len(low_roi_high_spend)
            
            return roi_analysis
        except Exception as e:
            logger.error(f"ROI analysis error: {e}")
            return {}

    def _detect_seasonal_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect seasonal patterns in the data"""
        try:
            if 'date_start' in data.columns:
                data['date_start'] = pd.to_datetime(data['date_start'])
                data['day_of_week'] = data['date_start'].dt.dayofweek
                data['month'] = data['date_start'].dt.month
                
                # Day of week patterns
                dow_patterns = data.groupby('day_of_week').agg({
                    'spend': 'mean',
                    'ctr': 'mean',
                    'clicks': 'mean'
                })
                
                # Monthly patterns
                monthly_patterns = data.groupby('month').agg({
                    'spend': 'mean',
                    'ctr': 'mean',
                    'clicks': 'mean'
                })
                
                return {
                    'day_of_week_patterns': dow_patterns.to_dict(),
                    'monthly_patterns': monthly_patterns.to_dict(),
                    'best_performing_day': dow_patterns['ctr'].idxmax(),
                    'best_performing_month': monthly_patterns['ctr'].idxmax()
                }
        except Exception as e:
            logger.error(f"Seasonal pattern detection error: {e}")
        return {}

    def _create_advanced_visualizations(self, data: pd.DataFrame, analysis_results: Dict) -> Dict[str, Any]:
        """Create advanced interactive visualizations"""
        try:
            viz_data = {}
            
            # Performance Overview Chart
            if 'spend' in data.columns and 'ctr' in data.columns:
                fig = px.scatter(data, x='spend', y='ctr', 
                               title='Performance vs Spend Analysis',
                               hover_data=['name', 'campaign_id'])
                viz_data['performance_scatter'] = fig.to_json()
            
            # Trend Analysis
            if 'date_start' in data.columns:
                trend_data = data.groupby('date_start').agg({
                    'spend': 'sum',
                    'impressions': 'sum',
                    'clicks': 'sum'
                }).reset_index()
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=trend_data['date_start'], y=trend_data['spend'], 
                                       name='Spend', mode='lines+markers'))
                fig.add_trace(go.Scatter(x=trend_data['date_start'], y=trend_data['clicks'], 
                                       name='Clicks', mode='lines+markers', yaxis='y2'))
                fig.update_layout(title='Performance Trends Over Time',
                                yaxis2=dict(overlaying='y', side='right'))
                viz_data['trend_analysis'] = fig.to_json()
            
            # Clustering Visualization
            if 'clusters' in analysis_results and 'clustered_data' in analysis_results['clusters']:
                cluster_df = pd.DataFrame(analysis_results['clusters']['clustered_data'])
                if 'cluster' in cluster_df.columns and 'spend' in cluster_df.columns and 'ctr' in cluster_df.columns:
                    fig = px.scatter(cluster_df, x='spend', y='ctr', color='cluster',
                                   title='Campaign Clusters Analysis')
                    viz_data['clustering_viz'] = fig.to_json()
            
            return viz_data
        except Exception as e:
            logger.error(f"Visualization error: {e}")
            return {}

    def _generate_recommendations(self, data: pd.DataFrame, analysis_results: Dict, user_query: str) -> List[Dict]:
        """Generate actionable recommendations"""
        recommendations = []
        
        try:
            # Performance-based recommendations
            if 'performance_scores' in analysis_results:
                score = analysis_results['performance_scores'].get('overall_performance', 0)
                if score < 50:
                    recommendations.append({
                        'type': 'performance_optimization',
                        'priority': 'high',
                        'title': 'Performance Optimization Needed',
                        'description': f'Overall performance score is {score:.1f}/100. Consider optimizing CTR and CPC.',
                        'action': 'Review and optimize underperforming campaigns'
                    })
            
            # Anomaly-based recommendations
            if 'anomalies' in analysis_results:
                anomaly_count = analysis_results['anomalies'].get('anomaly_count', 0)
                if anomaly_count > 0:
                    recommendations.append({
                        'type': 'anomaly_detection',
                        'priority': 'medium',
                        'title': f'{anomaly_count} Anomalies Detected',
                        'description': 'Unusual patterns detected in your data. Review for potential issues.',
                        'action': 'Investigate anomalous campaigns or ads'
                    })
            
            # ROI-based recommendations
            if 'roi_analysis' in analysis_results:
                roi_opps = analysis_results['roi_analysis'].get('optimization_opportunities', 0)
                if roi_opps > 0:
                    recommendations.append({
                        'type': 'roi_optimization',
                        'priority': 'high',
                        'title': f'{roi_opps} ROI Optimization Opportunities',
                        'description': 'High-spend campaigns with low ROI identified.',
                        'action': 'Review and optimize low-ROI campaigns'
                    })
            
            # Seasonal recommendations
            if 'seasonal_patterns' in analysis_results:
                best_day = analysis_results['seasonal_patterns'].get('best_performing_day')
                if best_day is not None:
                    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    recommendations.append({
                        'type': 'seasonal_optimization',
                        'priority': 'low',
                        'title': 'Seasonal Pattern Detected',
                        'description': f'{day_names[best_day]} shows the best performance.',
                        'action': 'Consider increasing budget on high-performing days'
                    })
            
        except Exception as e:
            logger.error(f"Recommendation generation error: {e}")
        
        return recommendations

    def _generate_advanced_insights(self, user_query: str, data: pd.DataFrame, sql_query: str, analysis_results: Dict) -> str:
        """Generate comprehensive AI insights"""
        try:
            if data is None or data.empty:
                return "No data found for the given query."

            # Create comprehensive analysis summary
            analysis_summary = self._create_advanced_analysis_summary(data, analysis_results)
            
            prompt = f"""
            You are an expert marketing analyst with deep knowledge of Facebook Ads optimization.
            
            User Query: {user_query}
            SQL Query: {sql_query}
            
            Data Analysis Summary:
            {analysis_summary}
            
            Provide comprehensive business insights including:
            1. Key performance indicators and trends
            2. Anomalies or unusual patterns detected
            3. Predictive insights and forecasts
            4. Optimization opportunities and recommendations
            5. Competitive positioning analysis
            6. ROI and efficiency insights
            7. Seasonal patterns and timing recommendations
            
            Focus on actionable insights that drive business value.
            Keep the response under 300 words but be comprehensive.
            """

            if hasattr(self, 'pro_model'):
                response = self.pro_model.generate_content(prompt)
                return response.text
            else:
                return "Advanced AI insights unavailable (Gemini Pro model not initialized)."

        except Exception as e:
            logger.error(f"Advanced insights generation error: {e}")
            return "Error generating advanced insights."

    def _create_advanced_analysis_summary(self, data: pd.DataFrame, analysis_results: Dict) -> str:
        """Create comprehensive analysis summary for AI insights"""
        summary = f"Dataset: {len(data)} records, {len(data.columns)} columns\n"
        
        # Basic statistics
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary += "Key Metrics:\n"
            for col in numeric_cols[:5]:
                if len(data) > 0:
                    summary += f"- {col}: mean={data[col].mean():.2f}, std={data[col].std():.2f}\n"
        
        # Analysis results
        if analysis_results:
            summary += "\nAdvanced Analysis Results:\n"
            
            if 'anomalies' in analysis_results:
                anomaly_count = analysis_results['anomalies'].get('anomaly_count', 0)
                summary += f"- Anomalies detected: {anomaly_count}\n"
            
            if 'performance_scores' in analysis_results:
                perf_score = analysis_results['performance_scores'].get('overall_performance', 0)
                summary += f"- Overall performance score: {perf_score:.1f}/100\n"
            
            if 'trends' in analysis_results:
                trends = analysis_results['trends']
                for metric, trend in trends.items():
                    direction = trend.get('direction', 'stable')
                    summary += f"- {metric}: {direction}\n"
            
            if 'clusters' in analysis_results:
                cluster_count = analysis_results['clusters'].get('cluster_count', 0)
                summary += f"- Data clustered into {cluster_count} groups\n"
        
        return summary

    def _update_learning_system(self, user_query: str, analysis_results: Dict):
        """Update the learning system with new insights"""
        try:
            # Update performance cache
            query_hash = hash(user_query)
            self.performance_cache[query_hash] = {
                'query': user_query,
                'timestamp': datetime.now(),
                'analysis_complexity': len(analysis_results),
                'success': True
            }
            
            # Save learned patterns periodically
            if len(self.query_history) % 10 == 0:  # Save every 10 queries
                self._save_learned_patterns()
                
        except Exception as e:
            logger.error(f"Learning system update error: {e}")

    def get_advanced_suggestions(self) -> List[Dict]:
        """Get advanced query suggestions with descriptions"""
        return [
            {
                'query': 'predict next week performance',
                'category': 'predictive',
                'description': 'AI-powered performance forecasting'
            },
            {
                'query': 'find optimization opportunities',
                'category': 'optimization',
                'description': 'Identify campaigns for improvement'
            },
            {
                'query': 'detect anomalies',
                'category': 'anomaly',
                'description': 'Find unusual patterns in data'
            },
            {
                'query': 'cluster analysis',
                'category': 'clustering',
                'description': 'Group similar campaigns together'
            },
            {
                'query': 'seasonal patterns',
                'category': 'seasonal',
                'description': 'Identify time-based patterns'
            },
            {
                'query': 'roi optimization',
                'category': 'roi',
                'description': 'Maximize return on ad spend'
            }
        ]

    # Keep the original methods for backward compatibility
    def process_query(self, user_query: str) -> Dict[str, Any]:
        """Backward compatibility method"""
        return self.process_advanced_query(user_query)
    
    def query(self, user_query: str) -> str:
        """Simple query method for backward compatibility"""
        try:
            result = self.process_advanced_query(user_query)
            
            if result.get('error'):
                return f"❌ Error: {result['error']}"
            
            # Format the response
            response = []
            
            # Add insights
            if result.get('insights'):
                response.append("## 📊 Analysis Results")
                response.append(result['insights'])
            
            # Add recommendations
            if result.get('recommendations'):
                response.append("\n## 💡 Recommendations")
                for rec in result['recommendations'][:3]:  # Show top 3
                    response.append(f"- **{rec['title']}**: {rec['description']}")
            
            # Add data summary
            if result.get('data') is not None and not result['data'].empty:
                response.append(f"\n## 📋 Data Summary")
                response.append(f"Found {len(result['data'])} records matching your query.")
                
                # Show sample data
                if len(result['data']) > 0:
                    response.append("\n**Sample data:**")
                    sample_data = result['data'].head(5).to_string()
                    response.append(f"```\n{sample_data}\n```")
            
            # If no structured response, try to provide a basic analysis
            if not response:
                if result.get('data') is not None:
                    if not result['data'].empty:
                        response.append("## 📊 Query Results")
                        response.append(f"Found {len(result['data'])} records.")
                        response.append("\n**Data Preview:**")
                        response.append(f"```\n{result['data'].head(10).to_string()}\n```")
                    else:
                        response.append("## 📊 Query Results")
                        response.append("No data found matching your query.")
                else:
                    response.append("## 📊 Query Results")
                    response.append("Unable to process your query. Please try a different question.")
            
            return "\n".join(response) if response else "No results found for your query."
            
        except Exception as e:
            logger.error(f"Error in simple query: {e}")
            return f"❌ Error processing query: {str(e)}"

    def _enhance_query(self, user_query: str) -> str:
        """Original query enhancement method"""
        enhanced = user_query.lower()
        
        # Expand abbreviations using synonyms
        for abbr, expansions in self.synonyms.items():
            for expansion in expansions:
                if abbr in enhanced:
                    enhanced = enhanced.replace(abbr, expansion)
        
        return enhanced

    def _initialize_query_patterns(self):
        """Initialize common query patterns and their SQL mappings"""
        self.query_patterns = {
            # Performance Metrics
            'performance': {
                'keywords': ['performance', 'metrics', 'kpis', 'results', 'stats'],
                'default_query': 'SELECT * FROM ads ORDER BY spend DESC LIMIT 10'
            },
            'ctr': {
                'keywords': ['ctr', 'click rate', 'click through', 'click-through'],
                'default_query': 'SELECT ai.ad_id, ai.ctr FROM ad_insights ai WHERE ai.ctr > 0 ORDER BY ai.ctr DESC'
            },
            'cpc': {
                'keywords': ['cpc', 'cost per click', 'click cost'],
                'default_query': 'SELECT ai.ad_id, ai.cpc FROM ad_insights ai WHERE ai.cpc > 0 ORDER BY ai.cpc ASC'
            },
            'cpm': {
                'keywords': ['cpm', 'cost per mille', 'impression cost'],
                'default_query': 'SELECT ai.ad_id, ai.cpm FROM ad_insights ai WHERE ai.cpm > 0 ORDER BY ai.cpm ASC'
            },
            'spend': {
                'keywords': ['spend', 'cost', 'budget', 'expense', 'money'],
                'default_query': 'SELECT ai.ad_id, ai.spend FROM ad_insights ai ORDER BY ai.spend DESC'
            },
            
            # Time-based Queries
            'recent': {
                'keywords': ['recent', 'latest', 'new', 'today', 'yesterday', 'this week'],
                'default_query': "SELECT * FROM ad_insights WHERE date(date_start) >= date('now', '-7 days')"
            },
            'monthly': {
                'keywords': ['month', 'monthly', 'this month', 'last month'],
                'default_query': "SELECT * FROM ad_insights WHERE strftime('%m', date_start) = strftime('%m', 'now')"
            },
            'weekly': {
                'keywords': ['week', 'weekly', 'this week', 'last week'],
                'default_query': "SELECT * FROM ad_insights WHERE date(date_start) >= date('now', '-7 days')"
            },
            
            # Status-based Queries
            'active': {
                'keywords': ['active', 'running', 'live', 'current'],
                'default_query': "SELECT * FROM ads WHERE status = 'ACTIVE'"
            },
            'paused': {
                'keywords': ['paused', 'stopped', 'inactive'],
                'default_query': "SELECT * FROM ads WHERE status = 'PAUSED'"
            },
            
            # Campaign Analysis
            'campaigns': {
                'keywords': ['campaign', 'campaigns'],
                'default_query': 'SELECT c.name, COUNT(ai.ad_id) as ad_count, SUM(ai.spend) as total_spend FROM campaigns c LEFT JOIN ad_insights ai ON c.id = ai.campaign_id GROUP BY c.id, c.name'
            },
            'top_campaigns': {
                'keywords': ['top campaign', 'best campaign', 'leading campaign'],
                'default_query': 'SELECT c.name, SUM(ai.spend) as total_spend, AVG(ai.ctr) as avg_ctr FROM campaigns c JOIN ad_insights ai ON c.id = ai.campaign_id GROUP BY c.id, c.name ORDER BY total_spend DESC LIMIT 10'
            },
            
            # Ad Set Analysis
            'adsets': {
                'keywords': ['adset', 'adsets', 'ad set'],
                'default_query': 'SELECT * FROM adsets ORDER BY created_time DESC'
            },
            
            # Comparative Analysis
            'compare': {
                'keywords': ['compare', 'vs', 'versus', 'difference', 'better', 'worse'],
                'default_query': 'SELECT ai.campaign_id, AVG(ai.ctr) as avg_ctr, AVG(ai.cpc) as avg_cpc FROM ad_insights ai GROUP BY ai.campaign_id ORDER BY avg_ctr DESC'
            },
            
            # Problem Detection
            'issues': {
                'keywords': ['problem', 'issue', 'low', 'high', 'poor', 'bad'],
                'default_query': 'SELECT * FROM ad_insights ai WHERE ai.ctr < 1.0 OR ai.cpc > 5.0 ORDER BY ai.spend DESC'
            },
            
            # Optimization
            'optimize': {
                'keywords': ['optimize', 'improve', 'boost', 'enhance'],
                'default_query': 'SELECT * FROM ad_insights ai WHERE ai.ctr < 1.0 AND ai.spend > 100 ORDER BY ai.spend DESC'
            }
        }

        # Common abbreviations and synonyms
        self.synonyms = {
            'ctr': ['click through rate', 'click rate', 'click-through rate'],
            'cpc': ['cost per click', 'click cost'],
            'cpm': ['cost per mille', 'cost per thousand impressions'],
            'roas': ['return on ad spend', 'roas'],
            'roi': ['return on investment', 'roi'],
            'spend': ['cost', 'expense', 'budget spent'],
            'impressions': ['views', 'displays', 'shows'],
            'clicks': ['clicks', 'click-throughs'],
            'reach': ['unique reach', 'unique users'],
            'frequency': ['avg frequency', 'frequency'],
            'campaign': ['campaign', 'ad campaign'],
            'adset': ['ad set', 'adset', 'ad group'],
            'ad': ['advertisement', 'ad', 'creative'],
            'active': ['running', 'live', 'active', 'enabled'],
            'paused': ['stopped', 'paused', 'inactive', 'disabled']
        }

    def _generate_advanced_sql_query(self, user_query: str) -> Optional[str]:
        """Generate advanced SQL query with enhanced context"""
        try:
            # First, try to generate a simple query for basic tables
            simple_query = self._generate_simple_query(user_query)
            if simple_query:
                return simple_query
            
            schema_context = self._create_schema_context()
            query_patterns_context = self._create_patterns_context()

            prompt = f"""
            You are an expert Facebook Ads data analyst with deep knowledge of marketing metrics and business intelligence. Convert natural language queries into precise SQL SELECT statements for a SQLite database.

            Database Schema:
            {schema_context}

            Common Query Patterns & Context:
            {query_patterns_context}

            Query Understanding Rules:
            1. **Short Queries**: For queries under 5 words, infer context from common patterns
               - "ctr" → "Show ads with click-through rates"
               - "spend" → "Show campaigns by total spend"
               - "active" → "Show active campaigns and ads"

            2. **Metric Focus**: When users mention metrics, include related performance data
               - CTR queries should include impressions and clicks
               - Spend queries should include ROI metrics
               - Performance queries should include multiple KPIs

            3. **Time Context**: Infer time ranges from context
               - "recent" = last 7 days
               - "this month" = current month
               - "latest" = last 30 days

            4. **Comparative Analysis**: When comparing, use appropriate aggregations
               - Campaign comparisons: GROUP BY campaign_id
               - Time comparisons: GROUP BY date ranges
               - Performance comparisons: ORDER BY relevant metrics

            5. **Business Intelligence**: Always include business-relevant context
               - Show campaign names with IDs
               - Include spend data when analyzing performance
               - Add status information for operational context

            SQL Generation Rules:
            - ONLY return a single SQL SELECT query - NO explanations, notes, or markdown
            - DO NOT include any text after the semicolon
            - Generate SIMPLE, VALID SQL - avoid complex nested expressions
            - Use JOINs to connect related tables (campaigns ↔ ad_insights ↔ campaign_insights)
            - Prefer INNER JOIN unless LEFT JOIN is needed for completeness
            - Use aggregations (SUM, AVG, COUNT, MAX, MIN) appropriately
            - Always include human-readable fields (names) alongside IDs
            - Use aliases for readability (c.name AS campaign_name)
            - Apply proper date filtering for timestamp fields
            - DO NOT use placeholder text like [Campaign ID] - use actual column names
            - Keep expressions simple and avoid deeply nested calculations

            Date Handling (SQLite):
            ✅ Campaign/Ad timestamps (created_time, start_time, stop_time):
               strftime('%m', created_time) = strftime('%m', 'now')
               date(created_time) >= date('now', '-30 days')
            ✅ Ad insights dates (date_start, date_stop):
               strftime('%m', date_start) = strftime('%m', 'now')
               date(date_start) >= date('now', '-30 days')
            
            IMPORTANT: Use SQLite functions only:
            - Use strftime() for date formatting: strftime('%Y-%m-%d', date_start)
            - Use date() for date arithmetic: date('now', '-7 days')
            - Use datetime() for datetime operations: datetime('now', '+1 day')
            - DO NOT use PostgreSQL functions like TO_DATE, NOW(), INTERVAL
            
            CRITICAL: Column Availability:
            - ad_insights table has: ctr, cpc, cpm, cpp, conversions, conversion_values (may be empty)
            - campaign_insights table has: spend, impressions, clicks, reach, conversions (may be empty)
            - If insights tables are empty, use basic tables (ads, campaigns, adsets) for structural analysis
            - For performance queries when insights unavailable, focus on status, budget, and metadata analysis

            User Query: {user_query}

            SQL Query:
            """

            if hasattr(self, 'text_model'):
                response = self.text_model.generate_content(prompt)
                return self._clean_sql_response(response.text)
            else:
                return None

        except Exception as e:
            logger.error(f"Error generating SQL query: {e}")
            return None

    def _generate_simple_query(self, user_query: str) -> Optional[str]:
        """Generate simple queries for basic table analysis when insights are not available"""
        query_lower = user_query.lower()
        
        # Enhanced pattern matching for meaningful analysis
        if any(word in query_lower for word in ['campaign', 'campaigns']):
            if 'count' in query_lower or 'how many' in query_lower:
                return """
                SELECT 
                    COUNT(*) as total_campaigns,
                    SUM(CASE WHEN status = 'ACTIVE' THEN 1 ELSE 0 END) as active_campaigns,
                    SUM(CASE WHEN status = 'PAUSED' THEN 1 ELSE 0 END) as paused_campaigns
                FROM campaigns;
                """
            elif 'active' in query_lower or 'running' in query_lower:
                return """
                SELECT 
                    c.id, 
                    c.name, 
                    c.status, 
                    c.objective,
                    c.daily_budget,
                    c.lifetime_budget,
                    COUNT(a.id) as ad_count,
                    COUNT(ads.id) as adset_count
                FROM campaigns c
                LEFT JOIN ads a ON c.id = a.campaign_id
                LEFT JOIN adsets ads ON c.id = ads.campaign_id
                WHERE c.status = 'ACTIVE'
                GROUP BY c.id, c.name, c.status, c.objective, c.daily_budget, c.lifetime_budget
                ORDER BY ad_count DESC;
                """
            elif 'status' in query_lower:
                return """
                SELECT 
                    status, 
                    COUNT(*) as count,
                    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM campaigns), 2) as percentage
                FROM campaigns 
                GROUP BY status 
                ORDER BY count DESC;
                """
            elif 'objective' in query_lower:
                return """
                SELECT 
                    objective, 
                    COUNT(*) as count,
                    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM campaigns), 2) as percentage
                FROM campaigns 
                WHERE objective IS NOT NULL
                GROUP BY objective 
                ORDER BY count DESC;
                """
            elif 'budget' in query_lower:
                return """
                SELECT 
                    name,
                    daily_budget,
                    lifetime_budget,
                    budget_remaining,
                    status,
                    objective
                FROM campaigns 
                WHERE daily_budget IS NOT NULL OR lifetime_budget IS NOT NULL
                ORDER BY COALESCE(daily_budget, lifetime_budget) DESC;
                """
            else:
                return """
                SELECT 
                    c.id, 
                    c.name, 
                    c.status, 
                    c.objective,
                    c.created_time,
                    COUNT(a.id) as ad_count,
                    COUNT(ads.id) as adset_count
                FROM campaigns c
                LEFT JOIN ads a ON c.id = a.campaign_id
                LEFT JOIN adsets ads ON c.id = ads.campaign_id
                GROUP BY c.id, c.name, c.status, c.objective, c.created_time
                ORDER BY c.created_time DESC 
                LIMIT 10;
                """
        
        elif any(word in query_lower for word in ['ad', 'ads']):
            if 'count' in query_lower or 'how many' in query_lower:
                return """
                SELECT 
                    COUNT(*) as total_ads,
                    SUM(CASE WHEN status = 'ACTIVE' THEN 1 ELSE 0 END) as active_ads,
                    SUM(CASE WHEN status = 'PAUSED' THEN 1 ELSE 0 END) as paused_ads
                FROM ads;
                """
            elif 'active' in query_lower or 'running' in query_lower:
                return """
                SELECT 
                    a.id, 
                    a.name, 
                    a.status, 
                    c.name as campaign_name,
                    a.created_time
                FROM ads a
                LEFT JOIN campaigns c ON a.campaign_id = c.id
                WHERE a.status = 'ACTIVE'
                ORDER BY a.created_time DESC;
                """
            elif 'status' in query_lower:
                return """
                SELECT 
                    status, 
                    COUNT(*) as count,
                    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM ads), 2) as percentage
                FROM ads 
                GROUP BY status 
                ORDER BY count DESC;
                """
            elif 'campaign' in query_lower:
                return """
                SELECT 
                    c.name as campaign_name,
                    COUNT(a.id) as ad_count,
                    SUM(CASE WHEN a.status = 'ACTIVE' THEN 1 ELSE 0 END) as active_ads,
                    SUM(CASE WHEN a.status = 'PAUSED' THEN 1 ELSE 0 END) as paused_ads
                FROM campaigns c
                LEFT JOIN ads a ON c.id = a.campaign_id
                GROUP BY c.id, c.name
                ORDER BY ad_count DESC;
                """
            else:
                return """
                SELECT 
                    a.id, 
                    a.name, 
                    a.status, 
                    c.name as campaign_name,
                    a.created_time
                FROM ads a
                LEFT JOIN campaigns c ON a.campaign_id = c.id
                ORDER BY a.created_time DESC 
                LIMIT 10;
                """
        
        elif any(word in query_lower for word in ['adset', 'adsets', 'ad set']):
            if 'count' in query_lower or 'how many' in query_lower:
                return """
                SELECT 
                    COUNT(*) as total_adsets,
                    SUM(CASE WHEN status = 'ACTIVE' THEN 1 ELSE 0 END) as active_adsets,
                    SUM(CASE WHEN status = 'PAUSED' THEN 1 ELSE 0 END) as paused_adsets
                FROM adsets;
                """
            elif 'active' in query_lower or 'running' in query_lower:
                return """
                SELECT 
                    ads.id, 
                    ads.name, 
                    ads.status, 
                    c.name as campaign_name,
                    ads.optimization_goal,
                    ads.daily_budget,
                    ads.created_time
                FROM adsets ads
                LEFT JOIN campaigns c ON ads.campaign_id = c.id
                WHERE ads.status = 'ACTIVE'
                ORDER BY ads.created_time DESC;
                """
            elif 'optimization' in query_lower or 'goal' in query_lower:
                return """
                SELECT 
                    optimization_goal, 
                    COUNT(*) as count,
                    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM adsets), 2) as percentage
                FROM adsets 
                WHERE optimization_goal IS NOT NULL
                GROUP BY optimization_goal 
                ORDER BY count DESC;
                """
            else:
                return """
                SELECT 
                    ads.id, 
                    ads.name, 
                    ads.status, 
                    c.name as campaign_name,
                    ads.optimization_goal,
                    ads.created_time
                FROM adsets ads
                LEFT JOIN campaigns c ON ads.campaign_id = c.id
                ORDER BY ads.created_time DESC 
                LIMIT 10;
                """
        
        elif 'budget' in query_lower:
            return """
            SELECT 
                c.name as campaign_name,
                c.daily_budget,
                c.lifetime_budget,
                c.budget_remaining,
                c.status,
                c.objective,
                COUNT(a.id) as ad_count,
                COUNT(ads.id) as adset_count
            FROM campaigns c
            LEFT JOIN ads a ON c.id = a.campaign_id
            LEFT JOIN adsets ads ON c.id = ads.campaign_id
            WHERE c.daily_budget IS NOT NULL OR c.lifetime_budget IS NOT NULL
            GROUP BY c.id, c.name, c.daily_budget, c.lifetime_budget, c.budget_remaining, c.status, c.objective
            ORDER BY COALESCE(c.daily_budget, c.lifetime_budget) DESC;
            """
        
        elif 'recent' in query_lower or 'latest' in query_lower:
            return """
            SELECT 
                'campaigns' as type, 
                id, 
                name, 
                status,
                created_time 
            FROM campaigns 
            WHERE created_time >= datetime('now', '-7 days')
            UNION ALL 
            SELECT 
                'ads' as type, 
                id, 
                name, 
                status,
                created_time 
            FROM ads 
            WHERE created_time >= datetime('now', '-7 days')
            ORDER BY created_time DESC;
            """
        
        elif 'overview' in query_lower or 'summary' in query_lower:
            return """
            SELECT 
                'campaigns' as table_name, 
                COUNT(*) as total_count,
                SUM(CASE WHEN status = 'ACTIVE' THEN 1 ELSE 0 END) as active_count,
                SUM(CASE WHEN status = 'PAUSED' THEN 1 ELSE 0 END) as paused_count
            FROM campaigns 
            UNION ALL 
            SELECT 
                'ads' as table_name, 
                COUNT(*) as total_count,
                SUM(CASE WHEN status = 'ACTIVE' THEN 1 ELSE 0 END) as active_count,
                SUM(CASE WHEN status = 'PAUSED' THEN 1 ELSE 0 END) as paused_count
            FROM ads 
            UNION ALL 
            SELECT 
                'adsets' as table_name, 
                COUNT(*) as total_count,
                SUM(CASE WHEN status = 'ACTIVE' THEN 1 ELSE 0 END) as active_count,
                SUM(CASE WHEN status = 'PAUSED' THEN 1 ELSE 0 END) as paused_count
            FROM adsets;
            """
        
        elif 'performance' in query_lower or 'analysis' in query_lower:
            return """
            SELECT 
                c.name as campaign_name,
                c.status as campaign_status,
                c.objective,
                COUNT(DISTINCT ads.id) as adset_count,
                COUNT(a.id) as ad_count,
                SUM(CASE WHEN a.status = 'ACTIVE' THEN 1 ELSE 0 END) as active_ads,
                SUM(CASE WHEN ads.status = 'ACTIVE' THEN 1 ELSE 0 END) as active_adsets,
                c.daily_budget,
                c.lifetime_budget
            FROM campaigns c
            LEFT JOIN adsets ads ON c.id = ads.campaign_id
            LEFT JOIN ads a ON c.id = a.campaign_id
            GROUP BY c.id, c.name, c.status, c.objective, c.daily_budget, c.lifetime_budget
            ORDER BY ad_count DESC;
            """
        
        return None

    def _create_patterns_context(self) -> str:
        """Create context for common query patterns"""
        context = "Common query patterns and their interpretations:\n"
        
        for pattern_name, pattern_info in self.query_patterns.items():
            keywords = ', '.join(pattern_info['keywords'])
            context += f"- {pattern_name.upper()}: Keywords [{keywords}] - {pattern_info['default_query']}\n"
        
        context += "\nSynonyms and abbreviations:\n"
        for abbr, expansions in self.synonyms.items():
            expansions_str = ', '.join(expansions)
            context += f"- {abbr} → {expansions_str}\n"
        
        return context

    def _create_schema_context(self) -> str:
        context = "Database Schema Information:\n"
        for table_name, table_info in self.schema_info.items():
            context += f"\nTable: {table_info['table']}\n"
            context += f"Description: {table_info['description']}\n"
            context += f"Columns: {', '.join(table_info['columns'])}\n"
        
        context += "\nIMPORTANT NOTES:\n"
        context += "- The insights tables (ad_insights, campaign_insights) may be empty if insights data hasn't been fetched.\n"
        context += "- For queries about performance metrics (spend, impressions, clicks, ctr, etc.), use the basic tables.\n"
        context += "- Focus on structural analysis: campaign counts, ad status, budget information.\n"
        context += "- Use JOINs to connect campaigns, adsets, and ads tables for comprehensive analysis.\n"
        
        return context

    def _clean_sql_response(self, response: str) -> str:
        # Remove markdown code blocks
        response = re.sub(r'```sql\n?', '', response)
        response = re.sub(r'```\n?', '', response)
        
        # Remove any explanatory text after the SQL query
        # Find the first semicolon and cut everything after it
        semicolon_index = response.find(';')
        if semicolon_index != -1:
            response = response[:semicolon_index + 1]
        
        # Remove any markdown formatting or notes
        response = re.sub(r'\*\*[^*]+\*\*', '', response)  # Remove **bold** text
        response = re.sub(r'\*[^*]+\*', '', response)      # Remove *italic* text
        response = re.sub(r'Note:.*', '', response, flags=re.DOTALL)  # Remove notes
        response = re.sub(r'This query.*', '', response, flags=re.DOTALL)  # Remove explanations
        
        # Remove placeholder text and fix malformed expressions
        response = re.sub(r'\[[^\]]+\]', 'NULL', response)  # Replace [placeholder] with NULL
        response = re.sub(r'CAST\([^)]+\)\s*\(', '(', response)  # Fix malformed CAST expressions
        response = re.sub(r'\)\s*\)\s*\)', ')', response)  # Fix multiple closing parentheses
        response = re.sub(r'\)\s*\)\s*\)\s*\)', ')', response)  # Fix even more parentheses
        
        response = response.strip()
        if not response.endswith(';'):
            response += ';'

        # Convert PostgreSQL functions to SQLite equivalents
        response = re.sub(r"TO_DATE\((date_start|date_stop),\s*'YYYY-MM-DD'\)", r"\1", response)
        response = re.sub(r"NOW\(\)", r"datetime('now')", response)
        response = re.sub(r"INTERVAL\s+'(\d+)\s+(day|week|month|year)s?'", r"'\1 \2s'", response)
        response = re.sub(r"DATE_ADD\(([^,]+),\s*INTERVAL\s+'([^']+)'\)", r"date(\1, '\2')", response)
        response = re.sub(r"DATE_SUB\(([^,]+),\s*INTERVAL\s+'([^']+)'\)", r"date(\1, '-\2')", response)
        
        # Fix any remaining PostgreSQL date functions
        response = re.sub(r"EXTRACT\(([^)]+)\s+FROM\s+([^)]+)\)", r"strftime('%\1', \2)", response)
        
        # Convert PostgreSQL date() function calls to SQLite format
        response = re.sub(r"date\('now',\s*'([^']+)'\)", r"date('now', '\1')", response)
        response = re.sub(r"date\(([^,]+),\s*'([^']+)'\)", r"date(\1, '\2')", response)

        return response

    def _extract_query_context(self, user_query: str) -> Dict[str, Any]:
        """Extract context from user query for learning"""
        context = {
            'query_length': len(user_query.split()),
            'contains_metrics': any(metric in user_query.lower() for metric in ['ctr', 'cpc', 'cpm', 'spend']),
            'contains_time': any(time_word in user_query.lower() for time_word in ['recent', 'month', 'week', 'today']),
            'contains_comparison': any(comp_word in user_query.lower() for comp_word in ['compare', 'vs', 'versus']),
            'query_type': 'simple' if len(user_query.split()) <= 3 else 'complex'
        }
        return context

    def _describe_cluster_characteristics(self, cluster_data: pd.DataFrame) -> str:
        """Describe the characteristics of a cluster"""
        try:
            characteristics = []
            
            if 'spend' in cluster_data.columns:
                avg_spend = cluster_data['spend'].mean()
                characteristics.append(f"Avg spend: ${avg_spend:.2f}")
            
            if 'ctr' in cluster_data.columns:
                avg_ctr = cluster_data['ctr'].mean()
                characteristics.append(f"Avg CTR: {avg_ctr:.2f}%")
            
            if 'status' in cluster_data.columns:
                status_counts = cluster_data['status'].value_counts()
                dominant_status = status_counts.index[0] if len(status_counts) > 0 else 'Unknown'
                characteristics.append(f"Dominant status: {dominant_status}")
            
            return ", ".join(characteristics)
        except Exception as e:
            return "Characteristics unavailable"

    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            'error': error_message,
            'sql_query': None,
            'data': None,
            'insights': None,
            'visualizations': None,
            'recommendations': None,
            'analysis_results': None,
            'enhanced_query': None
        }

    def get_suggested_queries(self) -> List[str]:
        """Get suggested queries for users based on available data"""
        return [
            "How many campaigns do I have?",
            "Show me active campaigns",
            "Campaign status overview",
            "Budget analysis",
            "Recent campaigns and ads",
            "Campaign performance analysis",
            "Ad set optimization goals",
            "Campaign objectives breakdown",
            "Active vs paused campaigns",
            "Campaign structure analysis",
            "Ad distribution by campaign",
            "Targeting and optimization overview"
        ]

# Note: GeminiQueryEngine class is defined above as a wrapper for AdvancedGeminiQueryEngine
