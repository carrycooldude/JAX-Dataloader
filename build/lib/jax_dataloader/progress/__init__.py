"""Progress tracking module for JAX applications."""

from typing import Optional, List, Callable
from tqdm import tqdm
import time

class ProgressTracker:
    """Tracks progress of data loading."""
    
    def __init__(
        self,
        total: int,
        desc: Optional[str] = None,
        unit: str = "it",
        leave: bool = True,
        update_interval: float = 0.1,
        callbacks: Optional[List[Callable[[float], None]]] = None,
        show_eta: bool = True,
    ):
        """Initialize the progress tracker.
        
        Args:
            total: Total number of items
            desc: Description of the progress
            unit: Unit of progress
            leave: Whether to leave the progress bar
            update_interval: Time interval between updates in seconds
            callbacks: List of callback functions to call on updates
            show_eta: Whether to show estimated time remaining
        """
        self.total = total
        self.current = 0
        self.start_time = time.time()
        self.update_interval = update_interval
        self.callbacks = callbacks or []
        self.show_eta = show_eta
        
        self.pbar = tqdm(
            total=total,
            desc=desc,
            unit=unit,
            leave=leave,
        )
        
    def update(self, n: int = 1):
        """Update the progress.
        
        Args:
            n: Number of items to update
        """
        self.current += n
        self.pbar.update(n)
        
        # Call callbacks if enough time has passed
        if time.time() - self.start_time >= self.update_interval:
            progress = self.get_progress()
            for callback in self.callbacks:
                callback(progress)
            self.start_time = time.time()
        
    def reset(self):
        """Reset the progress tracker."""
        self.current = 0
        self.start_time = time.time()
        self.pbar.reset()
        
    def get_progress(self) -> float:
        """Get the current progress as a fraction.
        
        Returns:
            Progress as a fraction between 0 and 1
        """
        return self.current / self.total if self.total > 0 else 0
        
    def get_eta(self) -> float:
        """Get the estimated time remaining.
        
        Returns:
            Estimated time remaining in seconds
        """
        if self.current == 0:
            return float('inf')
            
        elapsed = time.time() - self.start_time
        rate = self.current / elapsed if elapsed > 0 else 0
        remaining = self.total - self.current
        return remaining / rate if rate > 0 else float('inf')
        
    def close(self):
        """Close the progress bar."""
        self.pbar.close()
        
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
