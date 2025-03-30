"""Progress tracking module for JAX applications."""

from typing import Optional
from tqdm import tqdm

class ProgressTracker:
    """Tracks progress of data loading."""
    
    def __init__(
        self,
        total: int,
        desc: Optional[str] = None,
        unit: str = "it",
        leave: bool = True,
    ):
        """Initialize the progress tracker.
        
        Args:
            total: Total number of items
            desc: Description of the progress
            unit: Unit of progress
            leave: Whether to leave the progress bar
        """
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
        self.pbar.update(n)
        
    def close(self):
        """Close the progress bar."""
        self.pbar.close()
        
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
