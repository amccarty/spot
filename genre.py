import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
from collections import Counter
import ast
import numpy as np

st.set_page_config(
    page_title="Spotify Genre Network",
    page_icon="🕸️",
    layout="wide"
)

st.title("🕸️ Spotify Genre Network")
st.markdown("Explore how music genres connect through shared sub-genres and artist overlaps")

@st.cache_data
def load_artists():
    """Load artist data with genre information."""
    
    # Try local file first
    try:
        df = pd.read_csv("artists.csv")
        st.sidebar.success(f"✅ Loaded {len(df):,} artists from local file")
    except FileNotFoundError:
        try:
            url = "https://dagshub.com/amccarty/my-first-repo/raw/s3:/my-first-repo/archive/artists.csv"
            df = pd.read_csv(url)
            st.sidebar.success(f"✅ Loaded {len(df):,} artists from DagsHub")
        except Exception as e:
            st.error(f"Failed to load data: {e}")
            return None
    
    return df

@st.cache_data
def parse_genres(df):
    """Parse the genres column and extract all unique genres."""
    all_niche_genres = []
    artist_genres = []
    
    for idx, row in df.iterrows():
        main_genre = row['main_genre']
        genres_str = row['genres']
        
        # Parse the genres list
        try:
            if pd.isna(genres_str) or genres_str == '[]':
                niche_genres = []
            else:
                niche_genres = ast.literal_eval(genres_str)
        except:
            niche_genres = []
        
        all_niche_genres.extend(niche_genres)
        artist_genres.append({
            'name': row['name'],
            'main_genre': main_genre,
            'niche_genres': niche_genres,
            'followers': row['followers'],
            'popularity': row['popularity']
        })
    
    return artist_genres, Counter(all_niche_genres)

@st.cache_data
def build_genre_network(artist_genres, min_cooccurrence=10):
    """Build a network where genres are connected if they appear together on artists."""
    
    # Count co-occurrences of niche genres
    cooccurrence = Counter()
    genre_to_main = {}  # Map niche genre to most common main genre
    genre_main_counts = {}
    
    for artist in artist_genres:
        niche = artist['niche_genres']
        main = artist['main_genre']
        
        # Track which main genre each niche genre belongs to most often
        for g in niche:
            if g not in genre_main_counts:
                genre_main_counts[g] = Counter()
            genre_main_counts[g][main] += 1
        
        # Count co-occurrences
        for i, g1 in enumerate(niche):
            for g2 in niche[i+1:]:
                pair = tuple(sorted([g1, g2]))
                cooccurrence[pair] += 1
    
    # Determine primary main_genre for each niche genre
    for g, counts in genre_main_counts.items():
        genre_to_main[g] = counts.most_common(1)[0][0]
    
    # Build network
    G = nx.Graph()
    
    # Add edges for co-occurring genres
    for (g1, g2), count in cooccurrence.items():
        if count >= min_cooccurrence:
            if not G.has_node(g1):
                G.add_node(g1, main_genre=genre_to_main.get(g1, 'Unknown'))
            if not G.has_node(g2):
                G.add_node(g2, main_genre=genre_to_main.get(g2, 'Unknown'))
            G.add_edge(g1, g2, weight=count)
    
    return G

def plot_network(G, selected_main_genres=None):
    """Create an interactive plotly network visualization."""
    
    if len(G.nodes()) == 0:
        return None
    
    # Filter by main genre if specified
    if selected_main_genres:
        nodes_to_keep = [n for n in G.nodes() if G.nodes[n].get('main_genre') in selected_main_genres]
        G = G.subgraph(nodes_to_keep).copy()
    
    if len(G.nodes()) == 0:
        return None
    
    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Color map for main genres
    main_genres = list(set(nx.get_node_attributes(G, 'main_genre').values()))
    colors = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel
    color_map = {genre: colors[i % len(colors)] for i, genre in enumerate(sorted(main_genres))}
    
    # Create edge traces
    edge_x = []
    edge_y = []
    edge_weights = []
    
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_weights.append(edge[2].get('weight', 1))
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines',
        opacity=0.3
    )
    
    # Create node traces (one per main genre for legend)
    node_traces = []
    
    for main_genre in sorted(main_genres):
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        
        for node in G.nodes():
            if G.nodes[node].get('main_genre') == main_genre:
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                
                # Node size based on degree (connections)
                degree = G.degree(node)
                node_size.append(10 + degree * 2)
                
                # Hover text
                connections = list(G.neighbors(node))[:10]
                conn_text = ", ".join(connections)
                if len(list(G.neighbors(node))) > 10:
                    conn_text += "..."
                node_text.append(f"<b>{node}</b><br>Main: {main_genre}<br>Connections: {degree}<br>Related: {conn_text}")
        
        if node_x:  # Only add trace if there are nodes
            node_traces.append(go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=[n for n in G.nodes() if G.nodes[n].get('main_genre') == main_genre],
                textposition="top center",
                textfont=dict(size=8),
                hovertext=node_text,
                name=main_genre,
                marker=dict(
                    color=color_map[main_genre],
                    size=node_size,
                    line=dict(width=1, color='white')
                )
            ))
    
    # Create figure
    fig = go.Figure(data=[edge_trace] + node_traces)
    
    fig.update_layout(
        title="Genre Network (genres connected if they frequently appear together)",
        showlegend=True,
        legend=dict(title="Main Genre", yanchor="top", y=0.99, xanchor="left", x=0.01),
        hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=700,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

# Need plotly express for colors
import plotly.express as px

# Load data
df = load_artists()

if df is not None:
    # Parse genres
    artist_genres, genre_counts = parse_genres(df)
    
    # Sidebar controls
    st.sidebar.header("🎛️ Network Controls")
    
    min_cooccurrence = st.sidebar.slider(
        "Minimum co-occurrence threshold",
        min_value=5,
        max_value=50,
        value=15,
        help="Only show connections between genres that appear together on at least this many artists"
    )
    
    # Get available main genres
    all_main_genres = sorted(df['main_genre'].dropna().unique())
    
    selected_main_genres = st.sidebar.multiselect(
        "Filter by Main Genre",
        options=all_main_genres,
        default=all_main_genres,
        help="Show only niche genres that belong to these main genres"
    )
    
    # Build network
    G = build_genre_network(artist_genres, min_cooccurrence=min_cooccurrence)
    
    st.sidebar.markdown(f"**Network Stats:**")
    st.sidebar.markdown(f"- Nodes (genres): {G.number_of_nodes()}")
    st.sidebar.markdown(f"- Edges (connections): {G.number_of_edges()}")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🕸️ Network Graph", "📊 Genre Stats", "🔍 Explore"])
    
    with tab1:
        fig = plot_network(G, selected_main_genres)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Node size = number of connections. Hover for details. Drag to explore.")
        else:
            st.warning("No genres match the current filters. Try lowering the co-occurrence threshold.")
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Most Common Niche Genres")
            top_genres = genre_counts.most_common(30)
            genre_df = pd.DataFrame(top_genres, columns=['Genre', 'Artist Count'])
            
            fig_bar = px.bar(
                genre_df,
                x='Artist Count',
                y='Genre',
                orientation='h',
                title="Top 30 Niche Genres by Artist Count"
            )
            fig_bar.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            st.subheader("Main Genre Distribution")
            main_genre_counts = df['main_genre'].value_counts()
            
            fig_pie = px.pie(
                values=main_genre_counts.values,
                names=main_genre_counts.index,
                title="Artists by Main Genre"
            )
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
            
            st.subheader("Most Connected Genres")
            if G.number_of_nodes() > 0:
                degrees = dict(G.degree())
                top_connected = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:15]
                conn_df = pd.DataFrame(top_connected, columns=['Genre', 'Connections'])
                st.dataframe(conn_df, use_container_width=True)
    
    with tab3:
        st.subheader("Explore Genre Connections")
        
        # Genre search
        available_genres = sorted([n for n in G.nodes()])
        
        if available_genres:
            selected_genre = st.selectbox(
                "Select a genre to explore",
                options=available_genres
            )
            
            if selected_genre and G.has_node(selected_genre):
                neighbors = list(G.neighbors(selected_genre))
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**{selected_genre}**")
                    st.markdown(f"- Main genre: {G.nodes[selected_genre].get('main_genre', 'Unknown')}")
                    st.markdown(f"- Connected to {len(neighbors)} other genres")
                    
                    # Show connection strengths
                    connections = []
                    for neighbor in neighbors:
                        weight = G[selected_genre][neighbor].get('weight', 0)
                        connections.append({'Genre': neighbor, 'Shared Artists': weight})
                    
                    conn_df = pd.DataFrame(connections).sort_values('Shared Artists', ascending=False)
                    st.dataframe(conn_df.head(20), use_container_width=True)
                
                with col2:
                    # Mini network of just this genre and its neighbors
                    subgraph = G.subgraph([selected_genre] + neighbors).copy()
                    
                    pos = nx.spring_layout(subgraph, k=2, seed=42)
                    
                    edge_x, edge_y = [], []
                    for edge in subgraph.edges():
                        x0, y0 = pos[edge[0]]
                        x1, y1 = pos[edge[1]]
                        edge_x.extend([x0, x1, None])
                        edge_y.extend([y0, y1, None])
                    
                    node_x = [pos[n][0] for n in subgraph.nodes()]
                    node_y = [pos[n][1] for n in subgraph.nodes()]
                    node_colors = ['red' if n == selected_genre else 'lightblue' for n in subgraph.nodes()]
                    node_sizes = [30 if n == selected_genre else 15 for n in subgraph.nodes()]
                    
                    fig_sub = go.Figure()
                    fig_sub.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', 
                                                  line=dict(width=1, color='#888'), hoverinfo='none'))
                    fig_sub.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers+text',
                                                  marker=dict(size=node_sizes, color=node_colors),
                                                  text=list(subgraph.nodes()),
                                                  textposition='top center',
                                                  textfont=dict(size=9),
                                                  hoverinfo='text'))
                    fig_sub.update_layout(
                        title=f"Connections for '{selected_genre}'",
                        showlegend=False,
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=400
                    )
                    st.plotly_chart(fig_sub, use_container_width=True)
                
                # Show sample artists with this genre
                st.markdown(f"**Sample artists tagged with '{selected_genre}':**")
                artists_with_genre = [a for a in artist_genres if selected_genre in a['niche_genres']]
                artists_with_genre = sorted(artists_with_genre, key=lambda x: x['popularity'], reverse=True)[:10]
                
                artist_df = pd.DataFrame([{
                    'Artist': a['name'],
                    'Popularity': a['popularity'],
                    'Followers': f"{a['followers']:,}",
                    'Other Genres': ', '.join([g for g in a['niche_genres'] if g != selected_genre][:5])
                } for a in artists_with_genre])
                
                st.dataframe(artist_df, use_container_width=True)
        else:
            st.info("Adjust the co-occurrence threshold to see genres in the network.")

else:
    st.error("Unable to load data. Please check your data source.")

# Footer
st.markdown("---")
st.markdown(
    "Data source: [DagsHub - amccarty/my-first-repo]"
    "(https://dagshub.com/amccarty/my-first-repo) | "
    "Built with Streamlit"
)
