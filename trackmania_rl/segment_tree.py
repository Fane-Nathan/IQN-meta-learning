"""
Pure Python Segment Tree implementation for Prioritized Experience Replay.
Uses iterative (non-recursive) algorithms for robustness and speed.
"""

import numpy as np


class SumSegmentTree:
    """
    Sum Segment Tree for O(log n) priority sampling.
    
    Uses iterative algorithms instead of recursive to avoid:
    - Stack overflow on large buffers
    - Index out of bounds errors
    """
    
    def __init__(self, capacity: int):
        """
        Initialize the segment tree.
        
        Args:
            capacity: Maximum number of elements (must be power of 2 for simplicity,
                     but we'll handle non-power-of-2 by rounding up)
        """
        # Round up to next power of 2 for clean binary tree
        self._capacity = 1
        while self._capacity < capacity:
            self._capacity *= 2
        
        # Tree storage: index 1 is root, indices [capacity, 2*capacity) are leaves
        # Index 0 is unused
        self._tree = np.zeros(2 * self._capacity, dtype=np.float64)
    
    @property
    def capacity(self):
        return self._capacity
    
    def __setitem__(self, idx: int, value: float):
        """Set value at leaf index idx and propagate up."""
        if isinstance(idx, (int, np.integer)):
            idx = int(idx)
            if idx < 0 or idx >= self._capacity:
                return  # Silently ignore out-of-bounds
            
            # Leaf node position
            tree_idx = idx + self._capacity
            self._tree[tree_idx] = value
            
            # Propagate up to root
            tree_idx //= 2
            while tree_idx >= 1:
                self._tree[tree_idx] = self._tree[2 * tree_idx] + self._tree[2 * tree_idx + 1]
                tree_idx //= 2
        else:
            # Array of indices
            idx = np.asarray(idx)
            value = np.asarray(value)
            if len(value) == 1:
                value = np.full_like(idx, value[0], dtype=np.float64)
            for i, v in zip(idx, value):
                self[int(i)] = float(v)
    
    def __getitem__(self, idx):
        """Get value at leaf index idx."""
        if isinstance(idx, (int, np.integer)):
            idx = int(idx)
            if idx < 0 or idx >= self._capacity:
                return 0.0
            return self._tree[idx + self._capacity]
        else:
            # Array indexing
            idx = np.asarray(idx).astype(np.int64)
            idx = np.clip(idx, 0, self._capacity - 1)
            return self._tree[idx + self._capacity]
    
    def sum(self, start: int = 0, end: int = None) -> float:
        """
        Returns sum of elements in range [start, end).
        
        Args:
            start: Start index (inclusive)
            end: End index (exclusive), defaults to capacity
        
        Returns:
            Sum of arr[start:end]
        """
        if end is None:
            end = self._capacity
        
        # Clamp to valid range
        start = max(0, min(start, self._capacity))
        end = max(0, min(end, self._capacity))
        
        if start >= end:
            return 0.0
        
        # Convert to tree indices (leaves)
        start += self._capacity
        end += self._capacity
        
        # Iterative range sum
        result = 0.0
        while start < end:
            if start % 2 == 1:  # start is right child
                result += self._tree[start]
                start += 1
            if end % 2 == 1:    # end is right child
                end -= 1
                result += self._tree[end]
            start //= 2
            end //= 2
        
        return result
    
    def find_prefixsum_idx(self, prefixsum: float) -> int:
        """
        Find the highest index i such that sum(arr[0:i]) <= prefixsum.
        
        Used for sampling according to priorities.
        
        Args:
            prefixsum: Target prefix sum
        
        Returns:
            Index where cumulative sum exceeds prefixsum
        """
        # Handle edge cases
        total = self._tree[1]  # Root contains total sum
        if total <= 0:
            return 0
        
        # Clamp prefixsum
        prefixsum = max(0.0, min(prefixsum, total))
        
        # Start at root, descend to leaf
        idx = 1
        while idx < self._capacity:
            left_child = 2 * idx
            right_child = 2 * idx + 1
            
            left_sum = self._tree[left_child]
            
            if prefixsum < left_sum:
                idx = left_child
            else:
                prefixsum -= left_sum
                idx = right_child
        
        # Convert tree index to leaf index
        return idx - self._capacity
    
    def find_prefixsum_idx_vec(self, prefixsums: np.ndarray) -> np.ndarray:
        """Vectorized version of find_prefixsum_idx."""
        prefixsums = np.asarray(prefixsums)
        return np.array([self.find_prefixsum_idx(p) for p in prefixsums], dtype=np.int64)


class MinSegmentTree:
    """Min Segment Tree for O(log n) minimum queries."""
    
    def __init__(self, capacity: int):
        self._capacity = 1
        while self._capacity < capacity:
            self._capacity *= 2
        
        self._tree = np.full(2 * self._capacity, np.inf, dtype=np.float64)
    
    @property
    def capacity(self):
        return self._capacity
    
    def __setitem__(self, idx: int, value: float):
        if idx < 0 or idx >= self._capacity:
            return
        
        tree_idx = idx + self._capacity
        self._tree[tree_idx] = value
        
        tree_idx //= 2
        while tree_idx >= 1:
            self._tree[tree_idx] = min(self._tree[2 * tree_idx], self._tree[2 * tree_idx + 1])
            tree_idx //= 2
    
    def __getitem__(self, idx: int) -> float:
        if idx < 0 or idx >= self._capacity:
            return np.inf
        return self._tree[idx + self._capacity]
    
    def min(self, start: int = 0, end: int = None) -> float:
        """Returns min of elements in range [start, end)."""
        if end is None:
            end = self._capacity
        
        start = max(0, min(start, self._capacity))
        end = max(0, min(end, self._capacity))
        
        if start >= end:
            return np.inf
        
        start += self._capacity
        end += self._capacity
        
        result = np.inf
        while start < end:
            if start % 2 == 1:
                result = min(result, self._tree[start])
                start += 1
            if end % 2 == 1:
                end -= 1
                result = min(result, self._tree[end])
            start //= 2
            end //= 2
        
        return result
