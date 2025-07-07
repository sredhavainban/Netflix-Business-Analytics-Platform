#!/usr/bin/env python3
"""
Netflix Business Analytics Platform - Professional Version
Enterprise-level data analysis with comprehensive visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime, timedelta
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Professional styling
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

class NetflixProfessionalAnalytics:
    """Professional Netflix data analytics platform with comprehensive visualizations"""
    
    def __init__(self, csv_path='netflix_titles.csv'):
        """Initialize the analytics platform"""
        self.csv_path = csv_path
        self.df = None
        self.report_data = {}
        self.insights = []
        
        # Create output directories
        os.makedirs('business_reports', exist_ok=True)
        os.makedirs('strategic_insights', exist_ok=True)
        os.makedirs('professional_visuals', exist_ok=True)
        os.makedirs('interactive_dashboards', exist_ok=True)
        
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
    
    def create_comprehensive_visualizations(self):
        """Create comprehensive professional visualizations"""
        print("\n📊 Creating Professional Visualizations...")
        
        # 1. Content Overview Dashboard
        self._create_content_overview_dashboard()
        
        # 2. Geographic Analysis
        self._create_geographic_analysis()
        
        # 3. Temporal Trends Analysis
        self._create_temporal_analysis()
        
        # 4. Content Strategy Analysis
        self._create_content_strategy_analysis()
        
        # 5. Rating Distribution Analysis
        self._create_rating_analysis()
        
        # 6. Duration Analysis
        self._create_duration_analysis()
        
        # 7. Genre Analysis
        self._create_genre_analysis()
        
        print("✅ All professional visualizations created successfully")
    
    def _create_content_overview_dashboard(self):
        """Create comprehensive content overview dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Content Type Distribution', 'Content Growth Over Time', 
                           'Top Countries', 'Content Ratings Distribution'),
            specs=[[{"type": "pie"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Content type distribution
        content_types = self.df['type'].value_counts()
        fig.add_trace(
            go.Pie(labels=content_types.index, values=content_types.values, 
                   name="Content Types", hole=0.3),
            row=1, col=1
        )
        
        # Content growth over time
        valid_data = self.df.dropna(subset=['year_added'])
        yearly_counts = valid_data.groupby('year_added').size()
        fig.add_trace(
            go.Scatter(x=yearly_counts.index, y=yearly_counts.values, 
                      mode='lines+markers', name="Growth", line=dict(width=3)),
            row=1, col=2
        )
        
        # Top countries
        top_countries = self._get_top_countries(10)
        fig.add_trace(
            go.Bar(x=list(top_countries.values()), y=list(top_countries.keys()), 
                   orientation='h', name="Countries"),
            row=2, col=1
        )
        
        # Rating distribution
        rating_counts = self.df['rating'].value_counts().head(10)
        fig.add_trace(
            go.Bar(x=rating_counts.index, y=rating_counts.values, name="Ratings"),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Netflix Content Overview Dashboard", 
                         showlegend=False)
        fig.write_html('interactive_dashboards/content_overview.html')
        
        # Save static version
        plt.figure(figsize=(16, 12))
        plt.suptitle('Netflix Content Overview Dashboard', fontsize=16, fontweight='bold')
        
        # Content type pie chart
        plt.subplot(2, 2, 1)
        plt.pie(content_types.values, labels=content_types.index, autopct='%1.1f%%')
        plt.title('Content Type Distribution')
        
        # Growth line chart
        plt.subplot(2, 2, 2)
        plt.plot(yearly_counts.index, yearly_counts.values, marker='o', linewidth=3)
        plt.title('Content Growth Over Time')
        plt.xlabel('Year')
        plt.ylabel('Number of Titles')
        
        # Top countries bar chart
        plt.subplot(2, 2, 3)
        countries = list(top_countries.keys())
        counts = list(top_countries.values())
        plt.barh(range(len(countries)), counts)
        plt.yticks(range(len(countries)), countries)
        plt.title('Top 10 Countries')
        plt.xlabel('Number of Titles')
        
        # Rating distribution
        plt.subplot(2, 2, 4)
        rating_counts.plot(kind='bar')
        plt.title('Content Ratings Distribution')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('professional_visuals/content_overview_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_geographic_analysis(self):
        """Create comprehensive geographic analysis"""
        # Get country data
        country_counts = self._get_top_countries(15)
        
        # Create interactive map-style visualization
        fig = go.Figure(data=[
            go.Bar(x=list(country_counts.values()), 
                   y=list(country_counts.keys()),
                   orientation='h',
                   marker_color='rgb(55, 83, 109)')
        ])
        
        fig.update_layout(
            title="Geographic Distribution of Netflix Content",
            xaxis_title="Number of Titles",
            yaxis_title="Country",
            height=600,
            showlegend=False
        )
        
        fig.write_html('interactive_dashboards/geographic_analysis.html')
        
        # Create static version with enhanced styling
        plt.figure(figsize=(12, 10))
        countries = list(country_counts.keys())
        counts = list(country_counts.values())
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(countries)))
        bars = plt.barh(range(len(countries)), counts, color=colors)
        
        # Add value labels
        for i, (bar, count) in enumerate(zip(bars, counts)):
            plt.text(bar.get_width() + max(counts)*0.01, bar.get_y() + bar.get_height()/2, 
                     f'{count:,}', va='center', fontsize=10, fontweight='bold')
        
        plt.yticks(range(len(countries)), countries)
        plt.xlabel('Number of Titles', fontsize=12, fontweight='bold')
        plt.title('Geographic Distribution of Netflix Content', fontsize=16, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('professional_visuals/geographic_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_temporal_analysis(self):
        """Create comprehensive temporal analysis"""
        valid_data = self.df.dropna(subset=['year_added', 'month_added'])
        
        # Monthly trends heatmap
        monthly_data = valid_data.groupby(['year_added', 'month_added']).size().reset_index(name='count')
        pivot_data = monthly_data.pivot(index='year_added', columns='month_added', values='count').fillna(0)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale='Viridis',
            text=pivot_data.values.astype(int),
            texttemplate="%{text}",
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title="Content Addition Patterns by Year and Month",
            xaxis_title="Month",
            yaxis_title="Year",
            height=500
        )
        
        fig.write_html('interactive_dashboards/temporal_analysis.html')
        
        # Create static version
        plt.figure(figsize=(14, 8))
        sns.heatmap(pivot_data, annot=True, fmt='.0f', cmap='YlGnBu', cbar_kws={'label': 'Number of Titles'})
        plt.title('Content Addition Patterns by Year and Month', fontsize=16, fontweight='bold')
        plt.xlabel('Month', fontsize=12)
        plt.ylabel('Year', fontsize=12)
        
        plt.tight_layout()
        plt.savefig('professional_visuals/temporal_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_content_strategy_analysis(self):
        """Create content strategy analysis visualizations"""
        # Content type trends over time
        valid_data = self.df.dropna(subset=['year_added'])
        content_trends = valid_data.groupby(['year_added', 'type']).size().unstack(fill_value=0)
        
        fig = go.Figure()
        
        for content_type in content_trends.columns:
            fig.add_trace(go.Scatter(
                x=content_trends.index,
                y=content_trends[content_type],
                mode='lines+markers',
                name=content_type,
                line=dict(width=3)
            ))
        
        fig.update_layout(
            title="Content Type Trends Over Time",
            xaxis_title="Year",
            yaxis_title="Number of Titles",
            height=500
        )
        
        fig.write_html('interactive_dashboards/content_strategy_analysis.html')
        
        # Create static version
        plt.figure(figsize=(12, 8))
        for content_type in content_trends.columns:
            plt.plot(content_trends.index, content_trends[content_type], 
                    marker='o', linewidth=3, label=content_type)
        
        plt.title('Content Type Trends Over Time', fontsize=16, fontweight='bold')
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Number of Titles', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('professional_visuals/content_strategy_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_rating_analysis(self):
        """Create comprehensive rating analysis"""
        rating_counts = self.df['rating'].value_counts()
        
        # Create interactive rating distribution
        fig = go.Figure(data=[
            go.Bar(x=rating_counts.index, y=rating_counts.values,
                   marker_color='rgb(158,202,225)')
        ])
        
        fig.update_layout(
            title="Content Ratings Distribution",
            xaxis_title="Rating",
            yaxis_title="Number of Titles",
            height=500
        )
        
        fig.write_html('interactive_dashboards/rating_analysis.html')
        
        # Create static version with enhanced styling
        plt.figure(figsize=(12, 8))
        bars = plt.bar(range(len(rating_counts)), rating_counts.values, 
                       color=plt.cm.Set3(np.linspace(0, 1, len(rating_counts))))
        
        # Add percentage labels
        total = len(self.df)
        for i, (bar, count) in enumerate(zip(bars, rating_counts.values)):
            percentage = (count / total) * 100
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(rating_counts.values)*0.01,
                     f'{percentage:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(range(len(rating_counts)), rating_counts.index, rotation=45)
        plt.title('Content Ratings Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('Rating', fontsize=12)
        plt.ylabel('Number of Titles', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('professional_visuals/rating_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_duration_analysis(self):
        """Create comprehensive duration analysis"""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Movie Duration Distribution', 'TV Show Seasons Distribution'),
            specs=[[{"type": "histogram"}, {"type": "histogram"}]]
        )
        
        # Movie duration histogram
        movie_durations = self.movies_df['duration_min'].dropna()
        fig.add_trace(
            go.Histogram(x=movie_durations, nbinsx=30, name="Movies"),
            row=1, col=1
        )
        
        # TV show seasons histogram
        show_seasons = self.shows_df['duration_seasons'].dropna()
        fig.add_trace(
            go.Histogram(x=show_seasons, nbinsx=20, name="TV Shows"),
            row=1, col=2
        )
        
        fig.update_layout(height=500, title_text="Content Duration Analysis")
        fig.write_html('interactive_dashboards/duration_analysis.html')
        
        # Create static version
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Movie duration distribution
        ax1.hist(movie_durations, bins=30, alpha=0.7, color='#E50914', edgecolor='black')
        ax1.axvline(movie_durations.mean(), color='red', linestyle='--', 
                    label=f'Mean: {movie_durations.mean():.1f} min')
        ax1.set_title('Movie Duration Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Duration (minutes)')
        ax1.set_ylabel('Number of Movies')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # TV Show seasons distribution
        ax2.hist(show_seasons, bins=20, alpha=0.7, color='#564D4D', edgecolor='black')
        ax2.axvline(show_seasons.mean(), color='red', linestyle='--', 
                    label=f'Mean: {show_seasons.mean():.1f} seasons')
        ax2.set_title('TV Show Seasons Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Number of Seasons')
        ax2.set_ylabel('Number of TV Shows')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('professional_visuals/duration_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_genre_analysis(self):
        """Create comprehensive genre analysis"""
        # Get genre data
        genre_counts = self._get_top_genres(15)
        
        # Create interactive genre visualization
        fig = go.Figure(data=[
            go.Bar(x=list(genre_counts.values()), 
                   y=list(genre_counts.keys()),
                   orientation='h',
                   marker_color='rgb(26, 118, 255)')
        ])
        
        fig.update_layout(
            title="Top Genres on Netflix",
            xaxis_title="Number of Titles",
            yaxis_title="Genre",
            height=600
        )
        
        fig.write_html('interactive_dashboards/genre_analysis.html')
        
        # Create static version
        plt.figure(figsize=(12, 10))
        genres = list(genre_counts.keys())
        counts = list(genre_counts.values())
        
        colors = plt.cm.plasma(np.linspace(0, 1, len(genres)))
        bars = plt.barh(range(len(genres)), counts, color=colors)
        
        # Add value labels
        for i, (bar, count) in enumerate(zip(bars, counts)):
            plt.text(bar.get_width() + max(counts)*0.01, bar.get_y() + bar.get_height()/2, 
                     f'{count:,}', va='center', fontsize=10, fontweight='bold')
        
        plt.yticks(range(len(genres)), genres)
        plt.xlabel('Number of Titles', fontsize=12, fontweight='bold')
        plt.title('Top Genres on Netflix', fontsize=16, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('professional_visuals/genre_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
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
            'top_genres': self._get_top_genres(5)
        }
        
        self.report_data['executive_summary'] = summary
        self._save_executive_summary(summary)
        
        return summary
    
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
            f.write(f"Average TV Show Seasons: {summary['avg_tv_seasons']:.1f}\n\n")
            
            f.write("TOP COUNTRIES:\n")
            for country, count in summary['top_countries'].items():
                f.write(f"  • {country}: {count:,} titles\n")
            
            f.write("\nTOP GENRES:\n")
            for genre, count in summary['top_genres'].items():
                f.write(f"  • {genre}: {count:,} titles\n")
    
    def run_complete_analysis(self):
        """Run complete professional business analysis"""
        print("🚀 Netflix Professional Business Analytics Platform")
        print("=" * 60)
        
        # Load and preprocess data
        if not self.load_and_preprocess_data():
            return False
        
        # Generate comprehensive analysis
        self.generate_executive_summary()
        self.create_comprehensive_visualizations()
        
        print("\n✅ Professional Analysis Complete!")
        print(f"📁 Reports saved to: business_reports/")
        print(f"📊 Professional visualizations saved to: professional_visuals/")
        print(f"🖥️ Interactive dashboards saved to: interactive_dashboards/")
        print(f"💡 Strategic insights saved to: strategic_insights/")
        
        return True

def main():
    """Main function to run the professional business analytics platform"""
    analytics = NetflixProfessionalAnalytics()
    analytics.run_complete_analysis()

if __name__ == "__main__":
    main() 