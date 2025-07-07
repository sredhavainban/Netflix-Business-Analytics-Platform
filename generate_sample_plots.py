#!/usr/bin/env python3
"""
Netflix Trends Analyzer - Plot Generator
This script generates visualizations using the actual Netflix titles dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Create visuals directory
os.makedirs('visuals', exist_ok=True)

def load_netflix_data():
    """Load and preprocess the Netflix titles dataset"""
    print("📊 Loading Netflix titles dataset...")
    
    # Load the CSV file
    df = pd.read_csv('netflix_titles.csv')
    
    # Data preprocessing
    print("🔧 Preprocessing data...")
    
    # Convert 'date_added' to datetime
    df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
    
    # Fill missing values
    df['country'] = df['country'].fillna('Unknown')
    df['director'] = df['director'].fillna('Unknown')
    df['cast'] = df['cast'].fillna('Unknown')
    df['listed_in'] = df['listed_in'].fillna('Unknown')
    
    # Extract year from date_added
    df['year_added'] = df['date_added'].dt.year
    
    print(f"✅ Loaded {len(df)} titles from the dataset")
    return df

def plot_content_growth(df):
    """Plot content growth over years using real data"""
    print("📈 Creating content growth plot...")
    
    # Get content counts by year
    content_per_year = df.groupby('year_added').size().reset_index(name='count')
    content_per_year = content_per_year.dropna()
    
    plt.figure(figsize=(14, 8))
    plt.plot(content_per_year['year_added'], content_per_year['count'], 
             marker='o', linewidth=3, markersize=8)
    plt.fill_between(content_per_year['year_added'], content_per_year['count'], alpha=0.3)
    plt.title('Netflix Content Growth Over Years', fontsize=16, fontweight='bold')
    plt.xlabel('Year Added to Netflix', fontsize=12)
    plt.ylabel('Number of Titles Added', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add annotation for peak
    max_idx = content_per_year['count'].idxmax()
    max_year = content_per_year.loc[max_idx, 'year_added']
    max_count = content_per_year.loc[max_idx, 'count']
    plt.annotate(f'Peak: {max_count} titles in {int(max_year)}',
                 xy=(max_year, max_count),
                 xytext=(max_year-2, max_count+50),
                 arrowprops=dict(arrowstyle='->', color='red'),
                 fontsize=10, color='red')
    
    plt.tight_layout()
    plt.savefig('visuals/content_growth.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_top_countries(df):
    """Plot top countries by content count using real data"""
    print("🌍 Creating top countries plot...")
    
    # Count countries
    country_counts = Counter()
    df['country'].dropna().apply(lambda x: country_counts.update([c.strip() for c in x.split(',')]))
    
    # Get top 10 countries
    top_countries = country_counts.most_common(10)
    countries, counts = zip(*top_countries)
    
    plt.figure(figsize=(12, 10))
    bars = plt.barh(range(len(countries)), counts, 
                    color=sns.color_palette('viridis', len(countries)))
    plt.yticks(range(len(countries)), countries)
    plt.xlabel('Number of Titles', fontsize=12)
    plt.title('Top 10 Countries by Netflix Content Count', fontsize=16, fontweight='bold')
    
    # Add value labels
    for i, (bar, count) in enumerate(zip(bars, counts)):
        plt.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2, 
                 f'{count:,}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('visuals/top_countries.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_genre_distribution(df):
    """Plot genre distribution using real data"""
    print("🎭 Creating genre distribution plot...")
    
    # Count genres
    genre_counts = Counter()
    df['listed_in'].dropna().apply(lambda x: genre_counts.update([g.strip() for g in x.split(',')]))
    
    # Get top 10 genres
    top_genres = genre_counts.most_common(10)
    genres, counts = zip(*top_genres)
    
    plt.figure(figsize=(12, 10))
    bars = plt.barh(range(len(genres)), counts, 
                    color=sns.color_palette('plasma', len(genres)))
    plt.yticks(range(len(genres)), genres)
    plt.xlabel('Number of Titles', fontsize=12)
    plt.title('Top 10 Genres on Netflix', fontsize=16, fontweight='bold')
    
    # Add value labels
    for i, (bar, count) in enumerate(zip(bars, counts)):
        plt.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, 
                 f'{count:,}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('visuals/genre_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_rating_distribution(df):
    """Plot content rating distribution using real data"""
    print("🔞 Creating rating distribution plot...")
    
    # Count ratings
    rating_counts = df['rating'].value_counts()
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(rating_counts)), rating_counts.values, 
                    color=sns.color_palette('RdYlBu', len(rating_counts)))
    plt.yticks(range(len(rating_counts)), rating_counts.index)
    plt.xlabel('Number of Titles', fontsize=12)
    plt.title('Content Ratings Distribution on Netflix', fontsize=16, fontweight='bold')
    
    # Add value labels
    for i, (bar, count) in enumerate(zip(bars, rating_counts.values)):
        percentage = (count / len(df)) * 100
        plt.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2, 
                 f'{count:,} ({percentage:.1f}%)', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('visuals/rating_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_temporal_trends(df):
    """Plot temporal trends heatmap using real data"""
    print("📅 Creating temporal trends plot...")
    
    # Extract year and month from date_added
    df['year_added'] = df['date_added'].dt.year
    df['month_added'] = df['date_added'].dt.month
    
    # Create pivot table for heatmap
    temporal_data = df.groupby(['year_added', 'month_added']).size().reset_index(name='count')
    pivot_data = temporal_data.pivot(index='year_added', columns='month_added', values='count').fillna(0)
    
    plt.figure(figsize=(14, 8))
    sns.heatmap(pivot_data, cmap='YlGnBu', annot=True, fmt='.0f', 
                cbar_kws={'label': 'Number of Titles'})
    plt.title('Content Addition Patterns by Year and Month', fontsize=16, fontweight='bold')
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Year', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('visuals/temporal_trends.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_content_type_distribution(df):
    """Plot content type distribution (Movies vs TV Shows)"""
    print("📺 Creating content type distribution plot...")
    
    # Count content types
    type_counts = df['type'].value_counts()
    
    plt.figure(figsize=(10, 8))
    colors = ['#E50914', '#564D4D']  # Netflix red and dark gray
    plt.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%', 
            colors=colors, startangle=90)
    plt.title('Distribution of Content Types on Netflix', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('visuals/content_type_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_duration_analysis(df):
    """Plot duration analysis for movies and TV shows"""
    print("⏱️ Creating duration analysis plot...")
    
    # Separate movies and TV shows
    movies = df[df['type'] == 'Movie'].copy()
    shows = df[df['type'] == 'TV Show'].copy()
    
    # Extract duration values
    movies['duration_min'] = movies['duration'].str.extract(r'(\d+)').astype(float)
    shows['duration_seasons'] = shows['duration'].str.extract(r'(\d+)').astype(float)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Movie duration distribution
    ax1.hist(movies['duration_min'].dropna(), bins=30, alpha=0.7, color='#E50914')
    ax1.set_title('Movie Duration Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Duration (minutes)')
    ax1.set_ylabel('Number of Movies')
    ax1.axvline(movies['duration_min'].mean(), color='red', linestyle='--', 
                label=f'Mean: {movies["duration_min"].mean():.1f} min')
    ax1.legend()
    
    # TV Show seasons distribution
    ax2.hist(shows['duration_seasons'].dropna(), bins=20, alpha=0.7, color='#564D4D')
    ax2.set_title('TV Show Seasons Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Number of Seasons')
    ax2.set_ylabel('Number of TV Shows')
    ax2.axvline(shows['duration_seasons'].mean(), color='red', linestyle='--', 
                label=f'Mean: {shows["duration_seasons"].mean():.1f} seasons')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('visuals/duration_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to generate all plots using real Netflix data"""
    print("🎬 Netflix Trends Analyzer - Using Real Data")
    print("=" * 50)
    
    # Load the real Netflix data
    df = load_netflix_data()
    
    # Generate plots using real data
    plot_content_growth(df)
    plot_top_countries(df)
    plot_genre_distribution(df)
    plot_rating_distribution(df)
    plot_temporal_trends(df)
    plot_content_type_distribution(df)
    plot_duration_analysis(df)
    
    print("\n✅ All visualizations generated and saved to 'visuals/' folder!")
    print("📁 Files created:")
    for file in os.listdir('visuals'):
        if file.endswith('.png'):
            print(f"   • {file}")
    
    # Print some key statistics
    print("\n📊 Key Statistics:")
    print(f"   • Total titles: {len(df):,}")
    print(f"   • Movies: {len(df[df['type'] == 'Movie']):,}")
    print(f"   • TV Shows: {len(df[df['type'] == 'TV Show']):,}")
    print(f"   • Date range: {df['date_added'].min().year} - {df['date_added'].max().year}")

if __name__ == "__main__":
    main() 