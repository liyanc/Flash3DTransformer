# Background and Problem
We have `coord[N,d]` points scattering to `NB` buckets.
A hash function maps `coord[N,d]` into `bucket[N]` with values in range `[0, NB)`.
To build a linear table, we still need `offset[N]` with each values being unique within each bucket.

Instead of artificially enforcing order in `offset[N]`, we opt for deciding `offset[N]` by on-chip arbitration during runtime through incrementing atomic counters assigned to each bucket. Atomicity ensure uniqueness and contiguousness of `offset[N]`. So, for each bucket, we have in-bucket offsets uniquely running from 0 to `bucket_size`.

Counters are an `UInt32` array of shape `[NB]` inited at zero. A thread at `i` shall obtain their counter at `counter[bucket[i]]`. It competes the atomic counter by `atomicInc`, which gives the old value back to the thread while ensuring the counter in global memory being immediately incremented. The old value sets `offset[i]`.

The problem is that counters reside in global memory and `atomicInc` on global memory is expensive.

# Solution
We propose a three-stage committing scheme.

The first stage obtains local offsets.
We shard points `coord[N, d]` into `coord[n, d]` chunks into blocks.
Each block holds a local `counter_shmem[NB]` in shared memory, zero-inited.
We run each block in the same way to `atomicInc` through `counter_shmem`.
We end the first stage after obtaining `offset_shmem[n]` for each block.

The second stage submits each block's local `counter_shmem` to global memory `counter` by `atomicAdd`.
`atomicAdd` gives each block back old snapshots of `counter_global_old` before committing this block's `counter_shmem`.
At this point, we have globally unique counters.

The third stage readjust block-local `offset_shmem` by adding `counter_global_old` to ensure global uniqueness of `offset`.
A thread at `i-th` position gets `adj[i] = counter_global_old[bucket[i]]` and make `offset[i] = offset_shmem[i] + adj[i]`.
Finally, we write back all results back to global memory.