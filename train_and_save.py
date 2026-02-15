"""
Script to train models and save them for deployment.
This script should be run before deploying the API.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from models import JaccardSimilarityModel, BayesianModel, SocialBayesianMarkovModel
import warnings

warnings.filterwarnings('ignore')


def load_data():
    """Load Epinions data."""
    print("Loading data from epinions_data/epinions.txt...")
    
    reviews = []
    with open('epinions_data/epinions.txt', 'r', encoding='utf-8', errors='ignore') as f:
        next(f)  # Skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t')
            if len(parts) >= 6:
                try:
                    user = parts[0]
                    item = parts[1]
                    stars = int(parts[2])
                    paid = float(parts[3])
                    time = int(parts[4])
                    words = parts[5] if len(parts) > 5 else ""
                    
                    reviews.append({
                        'user': user,
                        'item': item,
                        'stars': stars,
                        'paid': paid,
                        'time': time,
                        'words': words
                    })
                except (ValueError, IndexError):
                    continue
    
    df = pd.DataFrame(reviews)
    print(f"Loaded {len(df)} reviews")
    return df


def load_trust_data():
    """Load trust network data."""
    print("Loading trust network...")
    
    trust_edges = []
    trust_file = 'epinions_data/network_trust.txt'
    
    try:
        with open(trust_file, 'r', encoding='utf-8', errors='ignore') as f:
            next(f)  # Skip header
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('\t')
                if len(parts) >= 2:
                    source = parts[0]
                    target = parts[1]
                    trust_edges.append({'source': source, 'target': target})
    except FileNotFoundError:
        print(f"Trust file not found: {trust_file}")
        return pd.DataFrame(columns=['source', 'target'])
    
    trust_df = pd.DataFrame(trust_edges)
    print(f"Loaded {len(trust_df)} trust relationships")
    return trust_df


def prepare_features(df):
    """Prepare features for Bayesian models."""
    # Basic features
    X = df[['paid', 'time', 'words']].copy()
    
    # Process words feature
    X['word_count'] = X['words'].apply(lambda x: len(str(x).split()))
    
    # Normalize time
    time_min = X['time'].min()
    time_max = X['time'].max()
    if time_max > time_min:
        X['time_normalized'] = (X['time'] - time_min) / (time_max - time_min + 1)
    else:
        X['time_normalized'] = 0.0
    
    # Select final features
    X_final = X[['paid', 'time_normalized', 'word_count']].values
    
    # Target: binary classification (high rating >= 4)
    y = (df['stars'] >= 4).astype(int).values
    
    return X_final, y


def prepare_social_features(df, trust_df):
    """Prepare features for Social Bayesian Markov model."""
    import networkx as nx
    
    # Build trust graph
    G = nx.DiGraph()
    for _, row in trust_df.iterrows():
        G.add_edge(row['source'], row['target'])
    
    # Compute network features
    pagerank = nx.pagerank(G, alpha=0.85) if len(G) > 0 else {}
    
    features = []
    for _, row in df.iterrows():
        user = row['user']
        
        # Basic features
        paid = row['paid']
        time = row['time']
        word_count = len(str(row['words']).split())
        
        # Social features
        trust_count = G.out_degree(user) if user in G else 0
        trustedby_count = G.in_degree(user) if user in G else 0
        pr = pagerank.get(user, 0.0)
        
        # Network features (simplified)
        clustering = nx.clustering(G.to_undirected(), user) if user in G else 0.0
        betweenness = 0.0  # Expensive to compute, use 0 for now
        
        # Derived features
        trust_strength = trust_count + trustedby_count
        social_influence = pr * trust_strength
        rating_consistency = 1.0  # Placeholder
        network_activity = trust_count
        trust_balance = trust_count / (trustedby_count + 1)
        avg_neighbor_rating = 3.0  # Placeholder
        network_centrality = pr
        trust_density = trust_count / (len(G.nodes()) + 1) if len(G) > 0 else 0
        review_effort = word_count * (1 + paid)
        
        # Log features
        log_trust_strength = np.log1p(trust_strength)
        log_social_influence = np.log1p(social_influence)
        log_network_activity = np.log1p(network_activity)
        log_word_count = np.log1p(word_count)
        log_paid = np.log1p(paid)
        
        # Squared features
        pagerank_squared = pr ** 2
        trust_count_squared = trust_count ** 2
        trustedby_count_squared = trustedby_count ** 2
        
        # Interaction features
        pagerank_trust = pr * trust_count
        pagerank_trustedby = pr * trustedby_count
        avg_rating_pagerank = avg_neighbor_rating * pr
        word_paid_interaction = word_count * paid
        
        # Ratio features
        trust_ratio = trust_count / (trust_strength + 1)
        trustedby_ratio = trustedby_count / (trust_strength + 1)
        
        # Combine all features (32 total)
        feature_vector = [
            paid, time, word_count,
            trust_count, trustedby_count, pr,
            clustering, betweenness,
            trust_strength, social_influence, rating_consistency, network_activity,
            trust_balance, avg_neighbor_rating, network_centrality, trust_density, review_effort,
            log_trust_strength, log_social_influence, log_network_activity,
            log_word_count, log_paid, pagerank_squared, trust_count_squared,
            trustedby_count_squared, pagerank_trust, pagerank_trustedby,
            avg_rating_pagerank, word_paid_interaction, trust_ratio, trustedby_ratio
        ]
        
        features.append(feature_vector)
    
    X = np.array(features)
    y = (df['stars'] >= 4).astype(int).values
    
    return X, y


def train_and_save_models(sample_size=None):
    """Train all models and save them."""
    # Create output directory
    output_dir = Path("saved_models")
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    df = load_data()
    trust_df = load_trust_data()
    
    # Sample data if requested (for faster training)
    if sample_size and len(df) > sample_size:
        print(f"\nSampling {sample_size} reviews for faster training...")
        df = df.sample(n=sample_size, random_state=42)
    
    # Split data
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    print(f"\nTrain set: {len(df_train)} reviews")
    print(f"Test set: {len(df_test)} reviews")
    
    # 1. Train Jaccard Similarity Model
    print("\n" + "="*60)
    print("TRAINING JACCARD SIMILARITY MODEL")
    print("="*60)
    jaccard_model = JaccardSimilarityModel(k=10)
    jaccard_model.fit(df_train)
    
    # Save model
    jaccard_path = output_dir / "jaccard_model.pkl"
    joblib.dump(jaccard_model, jaccard_path)
    print(f"\nSaved Jaccard model to {jaccard_path}")
    
    # 2. Train Bayesian Model
    print("\n" + "="*60)
    print("TRAINING BAYESIAN MODEL")
    print("="*60)
    X_train, y_train = prepare_features(df_train)
    bayesian_model = BayesianModel(alpha=1.0)
    bayesian_model.fit(X_train, y_train, df_train=df_train)
    
    # Save model
    bayesian_path = output_dir / "bayesian_model.pkl"
    joblib.dump(bayesian_model, bayesian_path)
    print(f"\nSaved Bayesian model to {bayesian_path}")
    
    # 3. Train Social Bayesian Markov Model
    print("\n" + "="*60)
    print("TRAINING SOCIAL BAYESIAN MARKOV MODEL")
    print("="*60)
    X_social_train, y_social_train = prepare_social_features(df_train, trust_df)
    social_model = SocialBayesianMarkovModel(alpha=1.0, mrf_weight=0.3)
    social_model.fit(X_social_train, y_social_train, df_train=df_train, trust_df=trust_df)
    
    # Save model
    social_path = output_dir / "social_bayesian_model.pkl"
    joblib.dump(social_model, social_path)
    print(f"\nSaved Social Bayesian Markov model to {social_path}")
    
    print("\n" + "="*60)
    print("ALL MODELS TRAINED AND SAVED SUCCESSFULLY!")
    print("="*60)
    print(f"\nModels saved in: {output_dir.absolute()}")
    print("\nYou can now deploy the API using:")
    print("  python app.py")
    print("  OR")
    print("  docker-compose up --build")


if __name__ == "__main__":
    import sys
    
    # Optional: specify sample size for faster training
    # Example: python train_and_save.py 10000
    sample_size = None
    if len(sys.argv) > 1:
        try:
            sample_size = int(sys.argv[1])
            print(f"Using sample size: {sample_size}")
        except ValueError:
            print("Invalid sample size, using all data")
    
    train_and_save_models(sample_size=sample_size)
