import time
import threading
from typing import Dict, Optional, Any
from queue import Queue
import json
import logging
from datetime import datetime
import psutil
import os
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.logging import RichHandler
from rich.table import Table
from rich.panel import Panel

class ProgressLogger:
    def __init__(self, log_file: Optional[str] = None, log_level: int = logging.INFO):
        self.console = Console()
        self.log_file = log_file
        self.log_level = log_level
        self._setup_logging()
        
        # Initialize metrics
        self.metrics: Dict[str, Any] = {
            'start_time': time.time(),
            'batches_processed': 0,
            'total_batches': 0,
            'current_epoch': 0,
            'total_epochs': 0,
            'memory_usage': [],
            'batch_times': [],
            'errors': []
        }
        
        # Setup progress display
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console
        )
        
        self._progress_task = None
        self._metrics_lock = threading.Lock()
        
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=self.log_level,
            format="%(message)s",
            handlers=[RichHandler(console=self.console)]
        )
        
        if self.log_file:
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            ))
            logging.getLogger().addHandler(file_handler)
            
    def start_epoch(self, epoch: int, total_epochs: int, total_batches: int):
        """Start tracking a new epoch"""
        with self._metrics_lock:
            self.metrics['current_epoch'] = epoch
            self.metrics['total_epochs'] = total_epochs
            self.metrics['total_batches'] = total_batches
            self.metrics['batches_processed'] = 0
            self.metrics['batch_times'] = []
            
        self._progress_task = self.progress.add_task(
            f"Epoch {epoch}/{total_epochs}",
            total=total_batches
        )
        
    def update_batch(self, batch_time: float, memory_usage: Optional[float] = None):
        """Update progress for a batch"""
        with self._metrics_lock:
            self.metrics['batches_processed'] += 1
            self.metrics['batch_times'].append(batch_time)
            
            if memory_usage is not None:
                self.metrics['memory_usage'].append(memory_usage)
                
        if self._progress_task is not None:
            self.progress.update(self._progress_task, advance=1)
            
    def log_error(self, error: Exception):
        """Log an error"""
        with self._metrics_lock:
            self.metrics['errors'].append({
                'time': datetime.now().isoformat(),
                'error': str(error),
                'type': type(error).__name__
            })
            
        logging.error(f"Error: {str(error)}")
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        with self._metrics_lock:
            metrics = self.metrics.copy()
            metrics['elapsed_time'] = time.time() - metrics['start_time']
            metrics['avg_batch_time'] = sum(metrics['batch_times']) / len(metrics['batch_times']) if metrics['batch_times'] else 0
            metrics['memory_usage_mb'] = [usage / (1024 * 1024) for usage in metrics['memory_usage']]
            return metrics
            
    def display_summary(self):
        """Display a summary of the training progress"""
        metrics = self.get_metrics()
        
        # Create summary table
        table = Table(title="Training Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Epochs", f"{metrics['current_epoch']}/{metrics['total_epochs']}")
        table.add_row("Batches Processed", str(metrics['batches_processed']))
        table.add_row("Total Time", f"{metrics['elapsed_time']:.2f}s")
        table.add_row("Average Batch Time", f"{metrics['avg_batch_time']:.4f}s")
        table.add_row("Memory Usage (avg)", f"{sum(metrics['memory_usage_mb']) / len(metrics['memory_usage_mb']):.2f}MB")
        
        # Display errors if any
        if metrics['errors']:
            error_table = Table(title="Errors")
            error_table.add_column("Time")
            error_table.add_column("Type")
            error_table.add_column("Message")
            
            for error in metrics['errors']:
                error_table.add_row(
                    error['time'],
                    error['type'],
                    error['error']
                )
                
            self.console.print(Panel(error_table))
            
        self.console.print(table)
        
    def save_metrics(self, file_path: str):
        """Save metrics to a file"""
        with open(file_path, 'w') as f:
            json.dump(self.get_metrics(), f, indent=2)
            
    def __del__(self):
        """Cleanup resources"""
        if self._progress_task is not None:
            self.progress.remove_task(self._progress_task) 