use std::sync::{Arc, Mutex};
use std::thread;
use std::collections::VecDeque;
use std::ptr;
use std::mem;
use rand::prelude::*;
use crossbeam::channel::{bounded, Receiver, Sender};
use crossbeam::deque::{Worker, Stealer, Injector};
use std::arch::x86_64::*;
use numa::{Node, NodeMask};
use memmap2::MmapOptions;

#[repr(C, align(64))]  // Cache line alignment
pub struct Loader {
    data: *mut u8,
    data_size: usize,
    batch_size: usize,
    num_workers: usize,
    prefetch_size: usize,
    indices: Vec<usize>,
    current_idx: usize,
    workers: Vec<thread::JoinHandle<()>>,
    batch_queue: Arc<Mutex<VecDeque<*mut u8>>>,
    stop_signal: Arc<Mutex<bool>>,
    work_queue: Arc<Injector<usize>>,
    worker_queues: Vec<Worker<usize>>,
    stealers: Vec<Stealer<usize>>,
}

#[inline(always)]
unsafe fn simd_copy_batch(src: *const f32, dst: *mut f32, size: usize) {
    let mut i = 0;
    while i + 8 <= size {
        let v = _mm256_loadu_ps(src.add(i));
        _mm256_storeu_ps(dst.add(i), v);
        i += 8;
    }
    while i < size {
        *dst.add(i) = *src.add(i);
        i += 1;
    }
}

#[no_mangle]
pub extern "C" fn create_loader(
    data: *mut u8,
    data_size: usize,
    batch_size: usize,
    shuffle: bool,
    num_workers: usize,
    prefetch_size: usize,
) -> *mut Loader {
    // Initialize NUMA topology
    let node = Node::current();
    let mut node_mask = NodeMask::new();
    node_mask.set(node.id());

    let mut indices: Vec<usize> = (0..data_size).collect();
    if shuffle {
        let mut rng = thread_rng();
        // Smart shuffling that maintains some locality
        for chunk in indices.chunks_mut(batch_size * 4) {
            chunk.shuffle(&mut rng);
        }
    }

    // Create work-stealing queue
    let work_queue = Arc::new(Injector::new());
    let mut worker_queues = Vec::with_capacity(num_workers);
    let mut stealers = Vec::with_capacity(num_workers);
    
    for _ in 0..num_workers {
        let worker = Worker::new_fifo();
        stealers.push(worker.stealer());
        worker_queues.push(worker);
    }

    let loader = Box::new(Loader {
        data,
        data_size,
        batch_size,
        num_workers,
        prefetch_size,
        indices,
        current_idx: 0,
        workers: Vec::new(),
        batch_queue: Arc::new(Mutex::new(VecDeque::with_capacity(prefetch_size))),
        stop_signal: Arc::new(Mutex::new(false)),
        work_queue: work_queue.clone(),
        worker_queues,
        stealers,
    });

    let loader_ptr = Box::into_raw(loader);
    unsafe {
        (*loader_ptr).start_workers();
    }
    loader_ptr
}

impl Loader {
    fn start_workers(&mut self) {
        let work_queue = self.work_queue.clone();
        let batch_queue = self.batch_queue.clone();
        let stop_signal = self.stop_signal.clone();
        let data = self.data;
        let batch_size = self.batch_size;
        let indices = self.indices.clone();

        for worker_id in 0..self.num_workers {
            let work_queue = work_queue.clone();
            let batch_queue = batch_queue.clone();
            let stop_signal = stop_signal.clone();
            let indices = indices.clone();
            let local_queue = self.worker_queues[worker_id].clone();
            let stealers = self.stealers.clone();

            let handle = thread::spawn(move || {
                // Pin thread to NUMA node
                let node = Node::current();
                node.set_cpu_affinity().unwrap();

                while !*stop_signal.lock().unwrap() {
                    // Try local queue first
                    let task = local_queue.pop().or_else(|| {
                        // Try stealing from other workers
                        stealers.iter().find_map(|stealer| stealer.steal().success())
                            .or_else(|| work_queue.steal().success())
                    });

                    if let Some(idx) = task {
                        unsafe {
                            // Allocate batch memory on current NUMA node
                            let batch_ptr = numa::alloc_local(batch_size * mem::size_of::<f32>()) as *mut f32;
                            
                            // Use SIMD for batch copying
                            for i in 0..batch_size {
                                let src = data.add(indices[idx + i] * mem::size_of::<f32>()) as *const f32;
                                let dst = batch_ptr.add(i);
                                simd_copy_batch(src, dst, 1);
                            }
                            
                            batch_queue.lock().unwrap().push_back(batch_ptr as *mut u8);
                        }
                    }
                }
            });

            self.workers.push(handle);
        }

        // Start prefetching
        thread::spawn(move || {
            let mut idx = 0;
            while idx < indices.len() {
                work_queue.push(idx);
                idx += batch_size;
            }
        });
    }
}

#[no_mangle]
pub extern "C" fn next_batch(loader: *mut Loader) -> *mut u8 {
    unsafe {
        if let Some(batch_ptr) = (*loader).batch_queue.lock().unwrap().pop_front() {
            batch_ptr
        } else {
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn free_loader(loader: *mut Loader) {
    unsafe {
        *(*loader).stop_signal.lock().unwrap() = true;
        for worker in (*loader).workers.drain(..) {
            worker.join().unwrap();
        }
        Box::from_raw(loader);
    }
} 