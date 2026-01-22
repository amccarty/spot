import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

st.set_page_config(
    page_title="Spotify Tempo Explorer",
    page_icon="🎵",
    layout="wide"
)

st.title("🎵 Spotify Tempo Distribution by Genre")
st.markdown("Explore how tempo (BPM) varies across different music genres")

@st.cache_data
def load_data():
    """Load song data - tries local file first, then DagsHub."""
    
    # Try local file first (for local development)
    try:
        df = pd.read_csv(
            "songs.csv",
            nrows=100000,
            usecols=[
                'name', 'artists', 'tempo', 'genre', 'year', 'popularity',
                'danceability', 'energy', 'valence', 'acousticness'
            ]
        )
        st.sidebar.success(f"✅ Loaded {len(df):,} songs from local file")
    except FileNotFoundError:
        # Fallback to DagsHub
        try:
            url = "https://dagshub.com/amccarty/my-first-repo/raw/s3:/my-first-repo/archive/songs.csv"
            chunks = pd.read_csv(url, chunksize=50000, usecols=[
                'name', 'artists', 'tempo', 'genre', 'year', 'popularity',
                'danceability', 'energy', 'valence', 'acousticness'
            ])
            
            df_list = []
            rows_loaded = 0
            for chunk in chunks:
                df_list.append(chunk)
                rows_loaded += len(chunk)
                if rows_loaded >= 100000:
                    break
            
            df = pd.concat(df_list, ignore_index=True)
            st.sidebar.success(f"✅ Loaded {len(df):,} songs from DagsHub")
        except Exception as e:
            st.error(f"Failed to load data: {e}")
            return None
    
    # Clean up tempo outliers (reasonable BPM range: 40-220)
    df = df[(df['tempo'] >= 40) & (df['tempo'] <= 220)]
    df = df.dropna(subset=['tempo', 'genre'])
    
    return df

# Load data
df = load_data()

if df is not None:
    # Sidebar filters
    st.sidebar.header("🎛️ Filters")
    
    # Genre filter
    all_genres = sorted(df['genre'].unique())
    selected_genres = st.sidebar.multiselect(
        "Select Genres",
        options=all_genres,
        default=all_genres[:6] if len(all_genres) >= 6 else all_genres
    )
    
    # Year range filter
    min_year = int(df['year'].min())
    max_year = int(df['year'].max())
    year_range = st.sidebar.slider(
        "Year Range",
        min_value=min_year,
        max_value=max_year,
        value=(2010, max_year)
    )
    
    # Filter data
    filtered_df = df[
        (df['genre'].isin(selected_genres)) &
        (df['year'] >= year_range[0]) &
        (df['year'] <= year_range[1])
    ]
    
    st.sidebar.markdown(f"**Showing:** {len(filtered_df):,} songs")
    
    # Main visualization tabs
    tab1, tab2, tab3 = st.tabs(["📊 Distribution", "📈 Trends", "🔍 Details"])
    
    with tab1:
        st.subheader("Tempo Distribution by Genre")
        
        if len(selected_genres) > 0:
            # Violin plot - shows distribution shape
            fig_violin = px.violin(
                filtered_df,
                x='genre',
                y='tempo',
                color='genre',
                box=True,
                points=False,
                title="Tempo (BPM) Distribution by Genre"
            )
            fig_violin.update_layout(
                xaxis_title="Genre",
                yaxis_title="Tempo (BPM)",
                showlegend=False,
                height=500
            )
            st.plotly_chart(fig_violin, use_container_width=True)
            
            # Stats table
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Genre Statistics**")
                stats = filtered_df.groupby('genre')['tempo'].agg([
                    ('Mean BPM', 'mean'),
                    ('Median BPM', 'median'),
                    ('Std Dev', 'std'),
                    ('Songs', 'count')
                ]).round(1)
                st.dataframe(stats, use_container_width=True)
            
            with col2:
                st.markdown("**Typical BPM Ranges**")
                # Box plot for cleaner range view
                fig_box = px.box(
                    filtered_df,
                    x='genre',
                    y='tempo',
                    color='genre',
                    title="BPM Range (Box Plot)"
                )
                fig_box.update_layout(showlegend=False, height=350)
                st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.warning("Please select at least one genre")
    
    with tab2:
        st.subheader("Tempo Trends Over Time")
        
        if len(selected_genres) > 0:
            # Average tempo by year and genre
            yearly_tempo = filtered_df.groupby(['year', 'genre'])['tempo'].mean().reset_index()
            
            fig_trend = px.line(
                yearly_tempo,
                x='year',
                y='tempo',
                color='genre',
                title="Average Tempo by Year and Genre",
                markers=True
            )
            fig_trend.update_layout(
                xaxis_title="Year",
                yaxis_title="Average Tempo (BPM)",
                height=500
            )
            st.plotly_chart(fig_trend, use_container_width=True)
            
            # Heatmap of tempo by year and genre
            pivot = filtered_df.pivot_table(
                values='tempo',
                index='genre',
                columns='year',
                aggfunc='mean'
            )
            
            fig_heatmap = px.imshow(
                pivot,
                title="Tempo Heatmap: Genre vs Year",
                labels=dict(x="Year", y="Genre", color="Avg BPM"),
                aspect="auto",
                color_continuous_scale="RdYlBu_r"
            )
            fig_heatmap.update_layout(height=400)
            st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.warning("Please select at least one genre")
    
    with tab3:
        st.subheader("Explore Individual Tracks")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Tempo range selector
            tempo_range = st.slider(
                "Filter by Tempo (BPM)",
                min_value=40,
                max_value=220,
                value=(100, 140)
            )
            
            genre_filter = st.selectbox(
                "Filter by Genre",
                options=["All"] + list(selected_genres)
            )
        
        with col2:
            # Filter for detail view
            detail_df = filtered_df[
                (filtered_df['tempo'] >= tempo_range[0]) &
                (filtered_df['tempo'] <= tempo_range[1])
            ]
            
            if genre_filter != "All":
                detail_df = detail_df[detail_df['genre'] == genre_filter]
            
            # Scatter plot: tempo vs other features
            fig_scatter = px.scatter(
                detail_df.sample(min(2000, len(detail_df))) if len(detail_df) > 0 else detail_df,
                x='tempo',
                y='energy',
                color='genre',
                size='popularity',
                hover_data=['name', 'artists'],
                title="Tempo vs Energy (size = popularity)",
                opacity=0.6
            )
            fig_scatter.update_layout(height=400)
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Sample tracks table
        st.markdown("**Sample Tracks in Selected Range:**")
        if len(detail_df) > 0:
            sample_tracks = detail_df.nlargest(20, 'popularity')[
                ['name', 'artists', 'genre', 'tempo', 'year', 'popularity']
            ].reset_index(drop=True)
            sample_tracks.columns = ['Track', 'Artist(s)', 'Genre', 'BPM', 'Year', 'Popularity']
            st.dataframe(sample_tracks, use_container_width=True)
        else:
            st.info("No tracks match the current filters")

else:
    st.error("Unable to load data. Please check your data source.")

# Footer
st.markdown("---")
st.markdown(
    "Data source: [DagsHub - amccarty/my-first-repo]"
    "(https://dagshub.com/amccarty/my-first-repo) | "
    "Built with Streamlit"
)
