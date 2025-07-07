#!/usr/bin/env python3
"""
Netflix Business Analytics Platform - Simplified Version
Enterprise-level data analysis for strategic decision making
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Professional styling
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

class NetflixBusinessAnalytics:
    """Enterprise-level Netflix data analytics platform"""
    
    def __init__(self, csv_path='netflix_titles.csv'):
        """Initialize the analytics platform"""
        self.csv_path = csv_path
        self.df = None
        self.report_data = {}
        self.insights = []
        
        # Create output directories
        os.makedirs('business_reports', exist_ok=True)
        os.makedirs('strategic_insights', exist_ok=True)
        os.makedirs('visuals', exist_ok=True)
        
    def load_and_preprocess_data(self):
        """Load and preprocess Netflix data with enterprise-level data quality checks"""
        print("🔍 Loading Netflix Business Data...")
        
        try:
            self.df = pd.read_csv(self.csv_path)
            print(f"✅ Successfully loaded {len(self.df):,} records")
            
            # Data quality assessment
            self._assess_data_quality()
            
            # Preprocessing
            self._preprocess_data()
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def _assess_data_quality(self):
        """Assess data quality and completeness"""
        print("\n📊 Data Quality Assessment:")
        
        # Missing values analysis
        missing_data = self.df.isnull().sum()
        missing_percentage = (missing_data / len(self.df)) * 100
        
        print(f"   • Total records: {len(self.df):,}")
        print(f"   • Missing values by column:")
        for col, missing in missing_data.items():
            if missing > 0:
                print(f"     - {col}: {missing:,} ({missing_percentage[col]:.1f}%)")
        
        # Data types and unique values
        print(f"   • Content types: {self.df['type'].value_counts().to_dict()}")
        
        # Check date range safely
        valid_dates = self.df['date_added'].dropna()
        if len(valid_dates) > 0:
            print(f"   • Date range: {valid_dates.min()} to {valid_dates.max()}")
        else:
            print("   • Date range: No valid dates found")
    
    def _preprocess_data(self):
        """Preprocess data for analysis"""
        print("\n🔧 Preprocessing data...")
        
        # Convert date_added to datetime
        self.df['date_added'] = pd.to_datetime(self.df['date_added'], errors='coerce')
        
        # Extract temporal features safely
        self.df['year_added'] = self.df['date_added'].dt.year
        self.df['month_added'] = self.df['date_added'].dt.month
        
        # Convert to numeric, handling NaN values
        self.df['year_added'] = pd.to_numeric(self.df['year_added'], errors='coerce')
        self.df['month_added'] = pd.to_numeric(self.df['month_added'], errors='coerce')
        
        # Handle missing values
        self.df['country'] = self.df['country'].fillna('Unknown')
        self.df['director'] = self.df['director'].fillna('Unknown')
        self.df['cast'] = self.df['cast'].fillna('Unknown')
        self.df['listed_in'] = self.df['listed_in'].fillna('Unknown')
        self.df['rating'] = self.df['rating'].fillna('Unknown')
        
        # Extract duration metrics
        self._extract_duration_metrics()
        
        print("✅ Data preprocessing completed")
    
    def _extract_duration_metrics(self):
        """Extract duration metrics for movies and TV shows"""
        # Movies duration
        movies = self.df[self.df['type'] == 'Movie'].copy()
        duration_extracted = movies['duration'].str.extract(r'(\d+)')
        movies['duration_min'] = pd.to_numeric(duration_extracted[0], errors='coerce')
        
        # TV Shows seasons
        shows = self.df[self.df['type'] == 'TV Show'].copy()
        duration_extracted = shows['duration'].str.extract(r'(\d+)')
        shows['duration_seasons'] = pd.to_numeric(duration_extracted[0], errors='coerce')
        
        # Store for analysis
        self.movies_df = movies
        self.shows_df = shows
    
    def generate_executive_summary(self):
        """Generate executive summary with key business metrics"""
        print("\n📈 Generating Executive Summary...")
        
        # Calculate average durations safely
        avg_movie_duration = self.movies_df['duration_min'].dropna().mean()
        avg_tv_seasons = self.shows_df['duration_seasons'].dropna().mean()
        
        # Handle date range safely
        valid_dates = self.df['date_added'].dropna()
        if len(valid_dates) > 0:
            date_range = f"{valid_dates.min().year} - {valid_dates.max().year}"
        else:
            date_range = "N/A"
        
        summary = {
            'total_content': len(self.df),
            'movies_count': len(self.df[self.df['type'] == 'Movie']),
            'tv_shows_count': len(self.df[self.df['type'] == 'TV Show']),
            'date_range': date_range,
            'avg_movie_duration': avg_movie_duration if pd.notna(avg_movie_duration) else 0,
            'avg_tv_seasons': avg_tv_seasons if pd.notna(avg_tv_seasons) else 0,
            'top_countries': self._get_top_countries(5),
            'top_genres': self._get_top_genres(5),
            'content_growth_rate': self._calculate_growth_rate()
        }
        
        self.report_data['executive_summary'] = summary
        self._save_executive_summary(summary)
        
        return summary
    
    def _get_top_countries(self, n=10):
        """Get top countries by content count"""
        country_counts = Counter()
        self.df['country'].dropna().apply(
            lambda x: country_counts.update([c.strip() for c in x.split(',')])
        )
        return dict(country_counts.most_common(n))
    
    def _get_top_genres(self, n=10):
        """Get top genres by content count"""
        genre_counts = Counter()
        self.df['listed_in'].dropna().apply(
            lambda x: genre_counts.update([g.strip() for g in x.split(',')])
        )
        return dict(genre_counts.most_common(n))
    
    def _calculate_growth_rate(self):
        """Calculate content growth rate"""
        # Filter out NaN values from year_added
        valid_years = self.df['year_added'].dropna()
        if len(valid_years) == 0:
            return 0
            
        yearly_counts = valid_years.groupby(valid_years).size()
        if len(yearly_counts) > 1:
            growth_rate = ((yearly_counts.iloc[-1] - yearly_counts.iloc[0]) / 
                          yearly_counts.iloc[0]) * 100
            return round(growth_rate, 2)
        return 0
    
    def _save_executive_summary(self, summary):
        """Save executive summary to file"""
        with open('business_reports/executive_summary.txt', 'w') as f:
            f.write("NETFLIX BUSINESS ANALYTICS - EXECUTIVE SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total Content Library: {summary['total_content']:,} titles\n")
            f.write(f"Movies: {summary['movies_count']:,}\n")
            f.write(f"TV Shows: {summary['tv_shows_count']:,}\n")
            f.write(f"Date Range: {summary['date_range']}\n")
            f.write(f"Average Movie Duration: {summary['avg_movie_duration']:.1f} minutes\n")
            f.write(f"Average TV Show Seasons: {summary['avg_tv_seasons']:.1f}\n")
            f.write(f"Content Growth Rate: {summary['content_growth_rate']}%\n\n")
            
            f.write("TOP COUNTRIES:\n")
            for country, count in summary['top_countries'].items():
                f.write(f"  • {country}: {count:,} titles\n")
            
            f.write("\nTOP GENRES:\n")
            for genre, count in summary['top_genres'].items():
                f.write(f"  • {genre}: {count:,} titles\n")
    
    def analyze_content_strategy(self):
        """Analyze content strategy and recommendations"""
        print("\n🎯 Analyzing Content Strategy...")
        
        insights = []
        
        # Content type analysis
        content_distribution = self.df['type'].value_counts(normalize=True) * 100
        insights.append(f"Content Mix: {content_distribution['Movie']:.1f}% Movies, {content_distribution['TV Show']:.1f}% TV Shows")
        
        # Genre analysis
        top_genres = self._get_top_genres(10)
        if top_genres:
            insights.append(f"Top Genre: {list(top_genres.keys())[0]} with {list(top_genres.values())[0]:,} titles")
        
        # Geographic analysis
        top_countries = self._get_top_countries(10)
        if top_countries:
            insights.append(f"Primary Market: {list(top_countries.keys())[0]} with {list(top_countries.values())[0]:,} titles")
        
        self.insights.extend(insights)
        self._save_strategic_insights(insights)
        
        return insights
    
    def _save_strategic_insights(self, insights):
        """Save strategic insights to file"""
        with open('strategic_insights/content_strategy_insights.txt', 'w') as f:
            f.write("STRATEGIC CONTENT INSIGHTS\n")
            f.write("=" * 30 + "\n\n")
            for i, insight in enumerate(insights, 1):
                f.write(f"{i}. {insight}\n")
    
    def generate_market_analysis(self):
        """Generate comprehensive market analysis"""
        print("\n🌍 Generating Market Analysis...")
        
        # Geographic distribution
        country_analysis = self._analyze_geographic_distribution()
        
        # Content rating analysis
        rating_analysis = self._analyze_content_ratings()
        
        # Save market analysis
        self._save_market_analysis(country_analysis, rating_analysis)
        
        return {
            'geographic': country_analysis,
            'ratings': rating_analysis
        }
    
    def _analyze_geographic_distribution(self):
        """Analyze geographic distribution of content"""
        country_counts = Counter()
        self.df['country'].dropna().apply(
            lambda x: country_counts.update([c.strip() for c in x.split(',')])
        )
        
        top_countries = dict(country_counts.most_common(15))
        total_content = sum(country_counts.values())
        
        # Calculate market concentration
        if total_content > 0:
            top_5_share = sum(list(top_countries.values())[:5]) / total_content * 100
        else:
            top_5_share = 0
        
        return {
            'top_countries': top_countries,
            'market_concentration': top_5_share,
            'total_markets': len(country_counts)
        }
    
    def _analyze_content_ratings(self):
        """Analyze content rating distribution"""
        rating_counts = self.df['rating'].value_counts()
        
        # Calculate family-friendly content percentage
        family_ratings = ['G', 'PG', 'PG-13', 'TV-Y', 'TV-Y7', 'TV-G', 'TV-PG']
        family_content = self.df[self.df['rating'].isin(family_ratings)]
        family_percentage = (len(family_content) / len(self.df)) * 100
        
        return {
            'rating_distribution': rating_counts.to_dict(),
            'family_content_percentage': family_percentage,
            'mature_content_percentage': 100 - family_percentage
        }
    
    def _save_market_analysis(self, geographic, ratings):
        """Save market analysis to file"""
        with open('business_reports/market_analysis.txt', 'w') as f:
            f.write("MARKET ANALYSIS REPORT\n")
            f.write("=" * 25 + "\n\n")
            
            f.write("GEOGRAPHIC DISTRIBUTION:\n")
            f.write(f"  • Total markets: {geographic['total_markets']}\n")
            f.write(f"  • Top 5 market concentration: {geographic['market_concentration']:.1f}%\n")
            f.write("  • Top countries:\n")
            for country, count in list(geographic['top_countries'].items())[:10]:
                f.write(f"    - {country}: {count:,} titles\n")
            
            f.write("\nCONTENT RATINGS:\n")
            f.write(f"  • Family-friendly content: {ratings['family_content_percentage']:.1f}%\n")
            f.write(f"  • Mature content: {ratings['mature_content_percentage']:.1f}%\n")
    
    def generate_business_recommendations(self):
        """Generate strategic business recommendations"""
        print("\n💡 Generating Business Recommendations...")
        
        recommendations = []
        
        # Content strategy recommendations
        content_distribution = self.df['type'].value_counts(normalize=True)
        if content_distribution['Movie'] > 0.7:
            recommendations.append("Consider increasing TV Show content to improve viewer retention")
        elif content_distribution['TV Show'] > 0.7:
            recommendations.append("Consider increasing Movie content for variety")
        
        # Geographic recommendations
        top_countries = self._get_top_countries(5)
        if top_countries and list(top_countries.values())[0] > sum(list(top_countries.values())[1:]) * 0.5:
            recommendations.append("Diversify geographic content to reduce market concentration risk")
        
        # Genre recommendations
        top_genres = self._get_top_genres(5)
        if top_genres and list(top_genres.values())[0] > sum(list(top_genres.values())[1:]) * 0.4:
            recommendations.append("Expand genre diversity to appeal to broader audience segments")
        
        self._save_recommendations(recommendations)
        return recommendations
    
    def _save_recommendations(self, recommendations):
        """Save business recommendations to file"""
        with open('business_reports/strategic_recommendations.txt', 'w') as f:
            f.write("STRATEGIC BUSINESS RECOMMENDATIONS\n")
            f.write("=" * 40 + "\n\n")
            for i, rec in enumerate(recommendations, 1):
                f.write(f"{i}. {rec}\n")
    
    def create_basic_visualizations(self):
        """Create basic visualizations"""
        print("\n📊 Creating Basic Visualizations...")
        
        # Content type distribution
        plt.figure(figsize=(10, 6))
        content_types = self.df['type'].value_counts()
        plt.pie(content_types.values, labels=content_types.index, autopct='%1.1f%%')
        plt.title('Content Type Distribution')
        plt.savefig('visuals/content_type_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Top countries
        top_countries = self._get_top_countries(10)
        plt.figure(figsize=(12, 8))
        countries = list(top_countries.keys())
        counts = list(top_countries.values())
        plt.barh(range(len(countries)), counts)
        plt.yticks(range(len(countries)), countries)
        plt.xlabel('Number of Titles')
        plt.title('Top 10 Countries by Content Count')
        plt.tight_layout()
        plt.savefig('visuals/top_countries.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Basic visualizations created")
    
    def run_complete_analysis(self):
        """Run complete business analysis"""
        print("🚀 Netflix Business Analytics Platform - Simplified Version")
        print("=" * 60)
        
        # Load and preprocess data
        if not self.load_and_preprocess_data():
            return False
        
        # Generate comprehensive analysis
        self.generate_executive_summary()
        self.analyze_content_strategy()
        self.generate_market_analysis()
        self.create_basic_visualizations()
        recommendations = self.generate_business_recommendations()
        
        print("\n✅ Analysis Complete!")
        print(f"📁 Reports saved to: business_reports/")
        print(f"📊 Visualizations saved to: visuals/")
        print(f"💡 Strategic insights saved to: strategic_insights/")
        
        return True

def main():
    """Main function to run the business analytics platform"""
    analytics = NetflixBusinessAnalytics()
    analytics.run_complete_analysis()

if __name__ == "__main__":
    main() 