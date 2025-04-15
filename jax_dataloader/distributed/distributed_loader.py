import jax
import jax.numpy as jnp
import numpy as np
from typing import Iterator, Union, Optional, Any, Dict
import threading
from queue import Queue
import time
from transformers import PreTrainedTokenizer
from datasets import Dataset
import ray
from ray.util.queue import Queue as RayQueue

class DistributedJAXDataLoader:
    def __init__(
        self,
        dataset: Union[Dataset, np.ndarray, jnp.ndarray],
        batch_size: int,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        num_workers: int = 2,
        prefetch_size: int = 2,
        use_mmap: bool = True,
        use_pinned_memory: bool = True,
        device: Optional[str] = None,
        world_size: int = 1,
        rank: int = 0
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.num_workers = num_workers
        self.prefetch_size = prefetch_size
        self.device = device
        self.world_size = world_size
        self.rank = rank
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()
            
        # Setup distributed queues
        self._batch_queue = RayQueue(maxsize=prefetch_size * 2)
        
        # Setup worker processes
        self._workers = []
        self._initialize_workers()
        
    def _initialize_workers(self):
        """Initialize distributed worker processes"""
        @ray.remote
        class DataWorker:
            def __init__(self, dataset, batch_size, tokenizer, rank, world_size):
                self.dataset = dataset
                self.batch_size = batch_size
                self.tokenizer = tokenizer
                self.rank = rank
                self.world_size = world_size
                
            def process_batch(self, indices):
                if isinstance(self.dataset, Dataset):
                    batch = self.dataset[indices]
                    if self.tokenizer:
                        batch = self.tokenizer(batch, padding=True, truncation=True, return_tensors="jax")
                else:
                    batch = self.dataset[indices]
                return batch
                
        # Create worker processes
        for i in range(self.num_workers):
            worker = DataWorker.remote(
                self.dataset,
                self.batch_size,
                self.tokenizer,
                self.rank,
                self.world_size
            )
            self._workers.append(worker)
            
    def _distributed_prefetch(self):
        """Distributed prefetch function"""
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(indices)
            
        # Split indices among workers
        worker_indices = np.array_split(indices, self.num_workers)
        
        # Process batches in parallel
        futures = []
        for worker, worker_idx in zip(self._workers, worker_indices):
            for i in range(0, len(worker_idx), self.batch_size):
                batch_idx = worker_idx[i:i + self.batch_size]
                future = worker.process_batch.remote(batch_idx)
                futures.append(future)
                
        # Collect results
        for future in futures:
            try:
                batch = ray.get(future)
                self._batch_queue.put(batch)
            except Exception as e:
                print(f"Error processing batch: {str(e)}")
                
    def __iter__(self):
        return self
        
    def __next__(self):
        try:
            batch = self._batch_queue.get(timeout=1.0)
            return batch
        except Exception:
            raise StopIteration
            
    def __del__(self):
        """Cleanup resources"""
        # Stop workers
        for worker in self._workers:
            ray.kill(worker)
            
        # Clear queue
        while not self._batch_queue.empty():
            try:
                self._batch_queue.get_nowait()
            except:
                break
                
        # Shutdown Ray if we initialized it
        if ray.is_initialized():
            ray.shutdown() 