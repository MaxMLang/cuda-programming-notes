### Streams and Concurrency Notes

#### The Core Concept: default-stream Behavior

* main point is understanding how GPU schedules tasks and how to run them concurrently.
* a CUDA Stream is just a queue of operations (kernels, memcopies) that the GPU runs in order.
* using multiple streams means multiple queues, so the GPU can run tasks from different ones at the same time.
* the key is the `--default-stream per-thread` compiler flag.

#### Task 1: stream_test.cu - Legacy vs. Per-Thread Streams

* Legacy Mode (`stream_legacy`):
    * default stream (stream 0) is a special blocking stream.
    * any task in the default stream waits for all other streams, and they wait for it.
    * we launched 8 kernels in 8 streams, but a dummy copy in the default stream in the loop made them run one by one.
    * time was slow: ~155 ms.

* Per-Thread Mode (`stream_per_thread`):
    * using the flag makes the default stream act like any other stream, it doesn't block.
    * the 8 kernels could now run at the same time.
    * time was much faster: ~19 ms (about 8x speedup).

#### Task 2: multithread_test.cu - CPU Threads and GPU Streams

* this used OpenMP to launch kernels from 8 CPU threads.
* Legacy Mode (`multithread_legacy`):
    * without the flag, all 8 CPU threads sent kernels to the same single default GPU stream.
    * GPU ran them one by one.
    * time was slow: ~141 ms.

* Per-Thread Mode (`multithread_per_thread`):
    * with the flag, each CPU thread got its own private default stream.
    * GPU saw 8 kernels in 8 streams and ran them all at once.
    * time was fast: ~18 ms.

#### Task 3: Overlapped Computation and Communication

* the `overlapped_processing.cu` file showed a practical use for streams.
* the idea is to make a pipeline to hide the time it takes to move data. While the GPU is processing chunk `n`, we use other streams to:
    1.  Copy results of chunk `n-1` from GPU back to CPU.
    2.  Copy input data for chunk `n+1` from CPU to GPU.
* this keeps the GPU busy, which is important for things like real-time video.

#### Task 4: Statistics Application - Parallel Bootstrapping

* `statistics_example.cu` showed how to use this for a real problem.
* Bootstrapping is a stats method where you make lots of random samples from a dataset.
* each sample is independent, so it's an "embarrassingly parallel" problem.
* the code launched a kernel in a separate stream for each of the 128 samples, so they all ran at the same time. Much faster than doing them sequentially.