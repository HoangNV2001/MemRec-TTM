"""
Base dataset class for recommendation systems
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


class _LazyNegatives(dict):
    """user_id -> np.ndarray of items the user never interacted with (computed on demand)"""
    
    def __init__(self, dataset):
        super().__init__()
        self._dataset = dataset
        self._all_items = np.arange(dataset.n_items)
    
    def _compute(self, user_id):
        negatives = np.setdiff1d(self._all_items, self._dataset.get_user_all_items(user_id))
        self[user_id] = negatives
        return negatives
    
    def __missing__(self, user_id):
        if user_id not in self._dataset.train_data:
            raise KeyError(user_id)
        return self._compute(user_id)
    
    def get(self, user_id, default=None):
        if dict.__contains__(self, user_id):
            return dict.__getitem__(self, user_id)
        if user_id in self._dataset.train_data:
            return self._compute(user_id)
        return default


class RecDataset:
    """Base recommendation dataset with leave-one-out split"""
    
    def __init__(self, data_path: str, seed: int = 42, holdout_users=None):
        """
        Args:
            data_path: Path to .inter file (TSV format)
            seed: Random seed
            holdout_users: If given, only these users get the leave-one-out split; every other user
                keeps all interactions in train (temporal cohorts: the graph sees all strictly-past events).
                None = upstream behaviour (every user with >=3 interactions is split, others dropped).
        """
        self.data_path = Path(data_path)
        self.seed = seed
        self.holdout_users = set(holdout_users) if holdout_users is not None else None
        self.rng = np.random.RandomState(seed)
        
        # Load data
        self.df = pd.read_csv(data_path, sep='\t')
        
        # Basic stats
        self.n_users = self.df['user_id'].max() + 1
        self.n_items = self.df['item_id'].max() + 1
        self.n_interactions = len(self.df)

        # Build user-item interaction dict, sorted by timestamp per user
        # (stable sort: ties keep file order; one groupby instead of a per-user filter)
        self._sort_by_timestamp()
        
        # Leave-one-out split
        self.train_data, self.valid_data, self.test_data = self._leave_one_out_split()
        
        # Precompute negative items for each user (HUGE speedup!)
        self._precompute_user_negatives()
        
        # Optional: Load additional data (for iAgent)
        self.instructions = None
        self.reviews = None
        self.item_metadata = None
        self.ranked_lists = None  # Pre-generated candidate lists from base recommender
        
    def _sort_by_timestamp(self):
        """Sort interactions by timestamp for each user (stable)"""
        df = self.df.sort_values(['user_id', 'timestamp'], kind='mergesort')
        users = df['user_id'].to_numpy()
        items = df['item_id'].to_numpy().astype(int)
        times = df['timestamp'].to_numpy()
        starts = np.flatnonzero(np.r_[True, users[1:] != users[:-1]])
        ends = np.r_[starts[1:], len(users)]
        self.user_items, self.user_times = {}, {}
        for a, b in zip(starts, ends):
            u = int(users[a])
            self.user_items[u] = items[a:b].tolist()
            self.user_times[u] = times[a:b].tolist()
    
    def _leave_one_out_split(self) -> Tuple[Dict, Dict, Dict]:
        """
        Leave-one-out split per user
        - Test: last item
        - Valid: second last item
        - Train: all remaining items
        
        Returns:
            train_data: {user_id: [item_ids]}
            valid_data: {user_id: item_id}
            test_data: {user_id: item_id}
        """
        train_data = {}
        valid_data = {}
        test_data = {}
        
        for user_id, items in self.user_items.items():
            if self.holdout_users is not None and user_id not in self.holdout_users:
                train_data[user_id] = items
                continue
            if len(items) < 3:
                # Skip users with too few interactions
                continue
            
            train_data[user_id] = items[:-2]
            valid_data[user_id] = items[-2]
            test_data[user_id] = items[-1]
        
        return train_data, valid_data, test_data
    
    def get_train_interactions(self) -> List[Tuple[int, int]]:
        """Get all training interactions as (user, item) pairs"""
        interactions = []
        for user_id, items in self.train_data.items():
            for item_id in items:
                interactions.append((user_id, item_id))
        return interactions
    
    def get_user_train_items(self, user_id: int) -> List[int]:
        """Get training items for a user"""
        return self.train_data.get(user_id, [])
    
    def get_user_all_items(self, user_id: int) -> List[int]:
        """Get all items interacted by a user (train + valid + test)"""
        items = self.train_data.get(user_id, []).copy()
        if user_id in self.valid_data:
            items.append(self.valid_data[user_id])
        if user_id in self.test_data:
            items.append(self.test_data[user_id])
        return items
    
    def get_user_history(self, user_id: int, split: str = 'test') -> List[int]:
        """
        Get user interaction history up to (and including) a specific split
        
        Args:
            user_id: User ID
            split: 'train', 'valid', or 'test'
            
        Returns:
            List of item IDs in chronological order
        """
        items = self.train_data.get(user_id, []).copy()
        
        if split in ['valid', 'test'] and user_id in self.valid_data:
            items.append(self.valid_data[user_id])
        
        if split == 'test' and user_id in self.test_data:
            items.append(self.test_data[user_id])
        
        return items
    
    def _precompute_user_negatives(self):
        """Negative items per user, computed lazily on first access.
        (Eager precompute needs n_users x n_items memory: ~11 GB on InstructRec Books.)"""
        self.user_negatives = _LazyNegatives(self)
    
    def _compute_item_popularity(self):
        """Compute item popularity from training data for popularity-based sampling"""
        from collections import Counter
        item_counts = Counter()
        
        # Count occurrences in training data only
        for user_id, items in self.train_data.items():
            for item in items:
                item_counts[item] += 1
        
        # Convert to numpy array aligned with item IDs
        self.item_popularity = np.zeros(self.n_items, dtype=np.float64)
        for item_id, count in item_counts.items():
            self.item_popularity[item_id] = count
        
        # Normalize to probabilities
        total = self.item_popularity.sum()
        if total > 0:
            self.item_popularity = self.item_popularity / total
        

    
    def sample_negative_items(self, user_id: int, n_samples: int, popularity_based: bool = False) -> List[int]:
        """
        Sample negative items for a user (items not interacted)
        
        Args:
            user_id: User ID
            n_samples: Number of negative items to sample
            popularity_based: If True, sample according to item popularity (harder evaluation)
                            If False, uniform random sampling (default, easier evaluation)
        """
        # Use precomputed negatives (MUCH faster!)
        all_negatives = self.user_negatives.get(user_id, np.arange(self.n_items))
        
        # If not enough negative items available, return all available + padding
        if len(all_negatives) < n_samples:
            # Use all available negatives + repeat last one to fill
            negatives = all_negatives.tolist()
            if negatives:
                negatives.extend([negatives[-1]] * (n_samples - len(negatives)))
            else:
                # Extreme case: user interacted with all items, use dummy negative
                negatives = [0] * n_samples
        else:
            if popularity_based and hasattr(self, 'item_popularity'):
                # Sample according to popularity (more realistic, harder evaluation)
                # Get popularity scores for negative items
                neg_popularity = self.item_popularity[all_negatives]
                
                # Normalize to probabilities
                if neg_popularity.sum() > 0:
                    neg_probs = neg_popularity / neg_popularity.sum()
                    negatives = self.rng.choice(all_negatives, size=n_samples, replace=False, p=neg_probs).tolist()
                else:
                    # Fallback to uniform if no popularity info
                    negatives = self.rng.choice(all_negatives, size=n_samples, replace=False).tolist()
            else:
                # Uniform random sampling (default)
                negatives = self.rng.choice(all_negatives, size=n_samples, replace=False).tolist()
        
        return negatives
    
    def get_stats(self) -> Dict:
        """Get dataset statistics"""
        return {
            'n_users': self.n_users,
            'n_items': self.n_items,
            'n_interactions': self.n_interactions,
            'n_train_users': len(self.train_data),
            'n_train_interactions': sum(len(items) for items in self.train_data.values()),
            'density': self.n_interactions / (self.n_users * self.n_items)
        }
    
    def __repr__(self):
        stats = self.get_stats()
        return (f"RecDataset(users={stats['n_users']}, items={stats['n_items']}, "
                f"interactions={stats['n_interactions']}, "
                f"density={stats['density']:.6f})")
    
    # ================== iAgent Support ==================
    
    def load_instructions(self):
        """Load user instructions and personas (for iAgent)"""
        instruction_path = self.data_path.parent / f"{self.data_path.stem}.instruction"
        if not instruction_path.exists():
            print(f"Warning: {instruction_path} not found")
            return
        
        df = pd.read_csv(instruction_path, sep='\t')
        self.instructions = {}
        for _, row in df.iterrows():
            self.instructions[int(row['user_id'])] = {
                'instruction': str(row['instruction']),
                'persona': str(row['persona']) if 'persona' in row else ''
            }
        print(f"Loaded instructions for {len(self.instructions)} users")
    
    def load_reviews(self):
        """Load user reviews (for iAgent)"""
        review_path = self.data_path.parent / f"{self.data_path.stem}.text"
        if not review_path.exists():
            print(f"Warning: {review_path} not found")
            return
        
        df = pd.read_csv(review_path, sep='\t')
        self.reviews = defaultdict(dict)
        for _, row in df.iterrows():
            user_id = int(row['user_id'])
            item_id = int(row['item_id'])
            review = str(row['review_text']) if pd.notna(row['review_text']) else ''
            self.reviews[user_id][item_id] = review
        print(f"Loaded reviews for {len(self.reviews)} users")
    
    def load_item_metadata(self):
        """Load item metadata (titles, descriptions) (for iAgent)"""
        meta_path = self.data_path.parent / f"{self.data_path.stem}.meta"
        if not meta_path.exists():
            print(f"Warning: {meta_path} not found")
            return
        
        self.item_metadata = {}
        # Read metadata in chunks to handle large files
        chunk_size = 10000
        for chunk in pd.read_csv(meta_path, sep='\t', chunksize=chunk_size):
            for _, row in chunk.iterrows():
                item_id = int(row['item_id'])
                self.item_metadata[item_id] = {
                    'asin': str(row['asin']) if 'asin' in row else '',
                    'title': str(row['title']) if 'title' in row else '',
                    'description': str(row['description']) if 'description' in row and pd.notna(row['description']) else ''
                }
        print(f"Loaded metadata for {len(self.item_metadata)} items")
    
    def load_ranked_lists(self):
        """
        Load pre-generated ranked lists (from base recommender)
        Format: user_id -> list of 10 candidate item_ids
        """
        import pickle
        
        # Try to load from .pkl file (original iAgent format)
        pkl_path = self.data_path.parent / f"{self.data_path.stem.replace('instructrec-', '')}All_recagent.pkl"
        
        if pkl_path.exists():
            print(f"Loading pre-generated ranked lists from {pkl_path}")
            try:
                df_pkl = pd.read_pickle(pkl_path)
                
                self.ranked_lists = {}
                # Map each user to their ranked_lists
                # Assuming df_pkl has 'reviewerID' and 'ranked_lists' columns
                for idx in range(len(df_pkl)):
                    # Get ranked_lists for this sample
                    ranked_list = df_pkl['ranked_lists'].iloc[idx]
                    
                    # Convert to list if it's numpy array
                    if hasattr(ranked_list, 'tolist'):
                        ranked_list = ranked_list.tolist()
                    else:
                        ranked_list = list(ranked_list)
                    
                    # Use index as user_id (assuming same order as .inter file)
                    # This is a simplification - in practice we'd need proper mapping
                    user_id = idx
                    self.ranked_lists[user_id] = ranked_list
                
                print(f"Loaded pre-generated ranked lists for {len(self.ranked_lists)} users")
                print(f"Example ranked_list length: {len(self.ranked_lists[0]) if 0 in self.ranked_lists else 'N/A'}")
                return
            except Exception as e:
                print(f"Error loading ranked_lists from pkl: {e}")
        
        print(f"Warning: {pkl_path} not found, ranked_lists not loaded")
    
    def load_all_iagent_data(self):
        """Load all data needed for iAgent"""
        self.load_instructions()
        self.load_reviews()
        self.load_item_metadata()
        self.load_ranked_lists()
    
    def get_user_history_text(self, user_id: int, items: List[int]) -> str:
        """
        Build user history string for iAgent
        
        Args:
            user_id: User ID
            items: List of item IDs in history
            
        Returns:
            Formatted history string
        """
        if self.item_metadata is None or self.reviews is None:
            return ""
        
        history_text = ""
        for item_id in items:
            # Get item info
            if item_id not in self.item_metadata:
                continue
            
            meta = self.item_metadata[item_id]
            title = meta['title'][:100] if len(meta['title']) > 100 else meta['title']
            desc = meta['description'][-200:] if len(meta['description']) > 200 else meta['description']
            
            # Clean description (remove HTML tags if any)
            import re
            desc = re.sub(r'<.*?>', '', desc)
            
            history_text += f"user historical information, item title:{title}, item description:{desc} ;"
        
        return history_text

