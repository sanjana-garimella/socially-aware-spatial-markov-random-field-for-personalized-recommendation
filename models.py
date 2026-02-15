"""
Recommendation Models for Epinions Dataset
Extracted from epinions_recommendation.ipynb for deployment
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from heapq import heappush, heappop
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
import networkx as nx
import warnings

warnings.filterwarnings('ignore')


class JaccardSimilarityModel:
    """
    Collaborative Filtering with Jaccard Similarity.
    Recommends items based on user-user and item-item similarity.
    """
    
    def __init__(self, k=10):
        """
        Initialize the model.
        
        Parameters:
        - k: Number of similar users/items to consider
        """
        self.k = k
        self.user_ratings = defaultdict(dict)
        self.item_users = defaultdict(set)
        self.user_similarities = {}
        self.item_similarities = {}
        self.all_items = set()
        
    def fit(self, df_train):
        """Fit the model on training data."""
        print("\nTraining Jaccard Similarity model...")
        
        # Build user-item matrix
        for _, row in df_train.iterrows():
            user = row['user']
            item = row['item']
            rating = row['stars']
            
            self.user_ratings[user][item] = rating
            self.item_users[item].add(user)
            self.all_items.add(item)
        
        # Compute user-user similarities
        print("  Computing user-user similarities...")
        users = list(self.user_ratings.keys())
        for i, user1 in enumerate(users):
            if i % 1000 == 0:
                print(f"    Progress: {i}/{len(users)} users")
            
            items1 = set(self.user_ratings[user1].keys())
            similarities = []
            
            for user2 in users:
                if user1 != user2:
                    items2 = set(self.user_ratings[user2].keys())
                    
                    intersection = len(items1 & items2)
                    union = len(items1 | items2)
                    
                    if union > 0:
                        similarity = intersection / union
                        similarities.append((user2, similarity))
            
            # Keep top-k similar users
            similarities.sort(key=lambda x: x[1], reverse=True)
            self.user_similarities[user1] = similarities[:self.k]
        
        # Compute item-item similarities
        print("  Computing item-item similarities...")
        items = list(self.item_users.keys())
        for i, item1 in enumerate(items):
            if i % 1000 == 0:
                print(f"    Progress: {i}/{len(items)} items")
            
            users1 = self.item_users[item1]
            similarities = []
            
            for item2 in items:
                if item1 != item2:
                    users2 = self.item_users[item2]
                    
                    intersection = len(users1 & users2)
                    union = len(users1 | users2)
                    
                    if union > 0:
                        similarity = intersection / union
                        similarities.append((item2, similarity))
            
            # Keep top-k similar items
            similarities.sort(key=lambda x: x[1], reverse=True)
            self.item_similarities[item1] = similarities[:self.k]
        
        print("  Model trained")
        return self
    
    def _score_item(self, user, item):
        """Score an item for a user based on similar users and items."""
        predictions = []
        weights = []
        
        # Method 1: Use similar users who reviewed this item
        if user in self.user_similarities:
            for similar_user, similarity in self.user_similarities[user]:
                if item in self.user_ratings[similar_user]:
                    rating = self.user_ratings[similar_user][item]
                    predictions.append(rating)
                    weights.append(similarity)
        
        # Method 2: Use similar items reviewed by this user
        if item in self.item_similarities:
            for similar_item, similarity in self.item_similarities[item]:
                if similar_item in self.user_ratings[user]:
                    rating = self.user_ratings[user][similar_item]
                    predictions.append(rating)
                    weights.append(similarity)
        
        # Weighted average
        if len(predictions) > 0:
            if sum(weights) > 0:
                return np.average(predictions, weights=weights)
            else:
                return np.mean(predictions)
        
        # Return neutral score if no similar users/items found
        return 3.0
    
    def predict_proba(self, df_test):
        """Predict probabilities for test set."""
        print("\nMaking predictions using Jaccard similarity...")
        predictions = []
        
        for idx, row in df_test.iterrows():
            user = row['user']
            item = row['item']
            score = self._score_item(user, item)
            predictions.append(score)
        
        # Convert ratings to probabilities (high rating >= 4.0)
        probs = 1 / (1 + np.exp(-(np.array(predictions) - 3.5)))
        return probs
    
    def predict_top_k(self, user, k=10):
        """Predict top-k item recommendations for a user."""
        user_reviewed_items = set(self.user_ratings[user].keys()) if user in self.user_ratings else set()
        candidate_items = self.all_items - user_reviewed_items
        
        # Use min-heap to keep only top-k items
        heap = []
        
        for item in candidate_items:
            score = self._score_item(user, item)
            
            if len(heap) < k:
                heappush(heap, (score, item))
            else:
                if score > heap[0][0]:
                    heappop(heap)
                    heappush(heap, (score, item))
        
        # Extract and sort recommendations
        recommendations = sorted(heap, key=lambda x: x[0], reverse=True)
        return [(item, score) for score, item in recommendations]


class BayesianModel:
    """
    Bayesian Model without popularity metrics.
    Uses only review features (paid, time, words).
    """
    
    def __init__(self, alpha=1.0):
        """
        Initialize the model.
        
        Parameters:
        - alpha: Prior strength for Bayesian inference
        """
        self.alpha = alpha
        self.scaler = StandardScaler()
        self.classifier = None
        self.user_items = defaultdict(set)
        self.all_items = set()
        self.time_min = 0
        self.time_max = 0
        
    def fit(self, X, y, df_train=None):
        """Fit the model."""
        print("\nTraining Bayesian model...")
        
        # Track user-item pairs for recommendations
        if df_train is not None:
            for _, row in df_train.iterrows():
                self.user_items[row['user']].add(row['item'])
                self.all_items.add(row['item'])
            
            # Store time range for normalization
            self.time_min = df_train['time'].min()
            self.time_max = df_train['time'].max()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train logistic regression
        print("  Training Logistic Regression classifier...")
        self.classifier = LogisticRegression(
            C=self.alpha,
            max_iter=1000,
            random_state=42,
            solver='lbfgs',
            n_jobs=-1,
            class_weight='balanced'
        )
        
        self.classifier.fit(X_scaled, y)
        print("  Model trained")
        
        return self
    
    def predict_proba(self, X):
        """Predict probabilities."""
        X_scaled = self.scaler.transform(X)
        probs = self.classifier.predict_proba(X_scaled)[:, 1]
        return probs
    
    def _score_item(self, user, item, paid=0.0, time=0, words=""):
        """Score an item for a user."""
        # Create feature vector
        word_count = len(str(words).split()) if words else 0
        
        # Normalize time
        if self.time_max > self.time_min:
            time_normalized = (time - self.time_min) / (self.time_max - self.time_min + 1)
        else:
            time_normalized = 0.0
        
        features = np.array([[paid, time_normalized, word_count]])
        prob = self.predict_proba(features)[0]
        
        return prob
    
    def predict_top_k(self, user, k=10, paid=0.0, time=0, words=""):
        """Predict top-k item recommendations for a user."""
        user_reviewed_items = self.user_items.get(user, set())
        candidate_items = self.all_items - user_reviewed_items
        
        # Use min-heap to keep only top-k items
        heap = []
        
        for item in candidate_items:
            score = self._score_item(user, item, paid=paid, time=time, words=words)
            
            if len(heap) < k:
                heappush(heap, (score, item))
            else:
                if score > heap[0][0]:
                    heappop(heap)
                    heappush(heap, (score, item))
        
        # Extract and sort recommendations
        recommendations = sorted(heap, key=lambda x: x[0], reverse=True)
        return [(item, score) for score, item in recommendations]


class SocialBayesianMarkovModel:
    """
    Social Bayesian Markov Model with MRF for social influence.
    Combines Bayesian inference with social network effects.
    """
    
    def __init__(self, alpha=1.0, mrf_weight=0.3):
        """
        Initialize the model.
        
        Parameters:
        - alpha: Prior strength for Bayesian inference
        - mrf_weight: Weight for MRF social influence (0-1)
        """
        self.alpha = alpha
        self.mrf_weight = mrf_weight
        self.scaler = StandardScaler()
        self.classifier = None
        self.trust_graph = nx.DiGraph()
        self.user_items = defaultdict(set)
        self.all_items = set()
        self.user_ratings_cache = defaultdict(dict)
        self.neighbor_predictions = defaultdict(dict)
        
    def fit(self, X, y, df_train=None, trust_df=None):
        """Fit the model with Bayesian approach and social network."""
        print("\nTraining Social Bayesian Markov model...")
        
        # Build trust graph
        if trust_df is not None:
            print("  Building trust graph...")
            for _, row in trust_df.iterrows():
                self.trust_graph.add_edge(row['source'], row['target'])
        
        # Track user-item pairs for recommendations
        if df_train is not None:
            for _, row in df_train.iterrows():
                self.user_items[row['user']].add(row['item'])
                self.all_items.add(row['item'])
                self.user_ratings_cache[row['user']][row['item']] = row['stars']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Base classifier: Enhanced Logistic Regression with cross-validation
        print("  Training enhanced Logistic Regression classifier...")
        self.classifier = LogisticRegressionCV(
            Cs=[0.1, 0.3, 0.5, 1.0, 2.0, 5.0],
            cv=3,
            max_iter=5000,
            random_state=42,
            solver='lbfgs',
            n_jobs=-1,
            class_weight='balanced',
            penalty='l2',
            tol=1e-6,
            scoring='roc_auc'
        )
        self.classifier.fit(X_scaled, y)
        print(f"  Classifier trained (best C: {self.classifier.C_[0]:.3f})")
        
        # Store training predictions for MRF
        print("  Computing neighbor predictions for MRF...")
        train_probs = self.classifier.predict_proba(X_scaled)[:, 1]
        
        if df_train is not None:
            for idx, row in df_train.iterrows():
                user = row['user']
                item = row['item']
                self.neighbor_predictions[user][item] = train_probs[idx]
        
        print("  Model trained")
        return self
    
    def predict_proba_base(self, X):
        """Predict probabilities using base classifier."""
        X_scaled = self.scaler.transform(X)
        probs = self.classifier.predict_proba(X_scaled)[:, 1]
        return probs
    
    def predict_proba_mrf(self, X, users, items):
        """Predict probabilities with MRF social influence."""
        base_probs = self.predict_proba_base(X)
        
        # Apply MRF smoothing
        mrf_probs = []
        for i, (user, item) in enumerate(zip(users, items)):
            base_prob = base_probs[i]
            
            # Get trusted neighbors' predictions
            if user in self.trust_graph:
                neighbors = list(self.trust_graph.successors(user))
                neighbor_preds = []
                
                for neighbor in neighbors:
                    if neighbor in self.neighbor_predictions and item in self.neighbor_predictions[neighbor]:
                        neighbor_preds.append(self.neighbor_predictions[neighbor][item])
                
                # MRF smoothing: weighted average with neighbors
                if len(neighbor_preds) > 0:
                    social_influence = np.mean(neighbor_preds)
                    final_prob = (1 - self.mrf_weight) * base_prob + self.mrf_weight * social_influence
                else:
                    final_prob = base_prob
            else:
                final_prob = base_prob
            
            mrf_probs.append(final_prob)
        
        return np.array(mrf_probs)
    
    def predict_top_k(self, user, k=10):
        """Predict top-k item recommendations for a user."""
        user_reviewed_items = self.user_items.get(user, set())
        candidate_items = self.all_items - user_reviewed_items
        
        # Use min-heap to keep only top-k items
        heap = []
        
        for item in candidate_items:
            # Create neutral feature vector with the exact training feature size.
            n_features = int(getattr(self.scaler, "n_features_in_", 31))
            base = [0.0, 0.5, 50.0]
            features = np.array([base + [0.0] * max(0, n_features - len(base))], dtype=float)
            base_prob = self.predict_proba_base(features)[0]
            
            # Apply MRF if user has trusted neighbors
            if user in self.trust_graph:
                neighbors = list(self.trust_graph.successors(user))
                neighbor_preds = []
                
                for neighbor in neighbors:
                    if neighbor in self.neighbor_predictions and item in self.neighbor_predictions[neighbor]:
                        neighbor_preds.append(self.neighbor_predictions[neighbor][item])
                
                if len(neighbor_preds) > 0:
                    social_influence = np.mean(neighbor_preds)
                    score = (1 - self.mrf_weight) * base_prob + self.mrf_weight * social_influence
                else:
                    score = base_prob
            else:
                score = base_prob
            
            if len(heap) < k:
                heappush(heap, (score, item))
            else:
                if score > heap[0][0]:
                    heappop(heap)
                    heappush(heap, (score, item))
        
        # Extract and sort recommendations
        recommendations = sorted(heap, key=lambda x: x[0], reverse=True)
        return [(item, score) for score, item in recommendations]
