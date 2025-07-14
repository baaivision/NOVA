# ------------------------------------------------------------------------
# Copyright (c) 2024-present, BAAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""Flex data loaders."""

import collections
import multiprocessing as mp
import time
import threading
import queue

import codewithgpu
import numpy as np

from diffnext.data.flex_pipelines import FeatureWorker


class BalancedQueues(object):
    """Balanced queues."""

    def __init__(self, base_queue, num=1):
        self.queues = [base_queue]
        self.queues += [mp.Queue(base_queue._maxsize) for _ in range(num - 1)]
        self.index = 0

    def put(self, obj, block=True, timeout=None):
        q = self.queues[self.index]
        q.put(obj, block=block, timeout=timeout)
        self.index = (self.index + 1) % len(self.queues)

    def get(self, block=True, timeout=None):
        q = self.queues[self.index]
        obj = q.get(block=block, timeout=timeout)
        self.index = (self.index + 1) % len(self.queues)
        return obj

    def get_n(self, num=1):
        outputs = []
        while len(outputs) < num:
            obj = self.get()
            if obj is not None:
                outputs.append(obj)
        return outputs


class DataLoaderBase(threading.Thread):
    """Base class of data loader."""

    def __init__(self, worker, **kwargs):
        super().__init__(daemon=True)
        self.seed = kwargs.pop("seed", 1337)
        self.shuffle = kwargs.pop("shuffle", True)
        self.shard_id = kwargs.get("shard_id", 0)
        self.num_shards = kwargs.get("num_shards", 1)
        self.batch_size = kwargs.get("batch_size", 1)
        self.num_workers = kwargs.get("num_workers", 1)
        self.queue_depth = kwargs.get("queue_depth", 2)
        # Build queues.
        self.reader_queue = mp.Queue(self.queue_depth * self.batch_size)
        self.worker_queue = mp.Queue(self.queue_depth * self.batch_size)
        self.batch_queue = queue.Queue(self.queue_depth)
        self.reader_queue = BalancedQueues(self.reader_queue, self.num_workers)
        self.worker_queue = BalancedQueues(self.worker_queue, self.num_workers)
        # Build readers.
        self.readers = [
            codewithgpu.DatasetReader(
                output_queue=self.reader_queue,
                partition_id=self.shard_id,
                num_partitions=self.num_shards,
                seed=self.seed + self.shard_id,
                shuffle=self.shuffle,
                **kwargs,
            )
        ]
        self.readers[0].start()
        time.sleep(0.1)
        # Build workers.
        self.workers = []
        for i in range(self.num_workers):
            p = worker()
            p.seed = self.seed + i + self.shard_id * self.num_workers
            p.reader_queue = self.reader_queue.queues[i]
            p.worker_queue = self.worker_queue.queues[i]
            p.start()
            self.workers.append(p)
            time.sleep(0.1)

        # Register cleanup callbacks.
        def cleanup():
            def terminate(processes):
                for p in processes:
                    p.terminate()
                    p.join()

            terminate(self.workers)
            terminate(self.readers)

        import atexit

        atexit.register(cleanup)
        # Start batch prefetching.
        self.start()

    def next(self):
        """Return the next batch of data."""
        return self.__next__()

    def run(self):
        """Main loop."""

    def __call__(self):
        return self.next()

    def __iter__(self):
        """Return the iterator self."""
        return self

    def __next__(self):
        """Return the next batch of data."""
        return [self.batch_queue.get()]


class DataLoader(DataLoaderBase):
    """Loader to return the batch of data."""

    def __init__(self, dataset, worker, **kwargs):
        kwargs.update({"path": dataset})  # Alias for codewithgpu.
        self.contiguous = kwargs.pop("contiguous", True)
        self.prefetch_count = kwargs.pop("prefetch_count", 50)
        super().__init__(worker, **kwargs)

    def run(self):
        """Main loop."""
        prev_inputs = self.worker_queue.get_n(self.prefetch_count * self.batch_size)
        next_inputs = []
        while True:
            # Use cached buffer for next N inputs.
            if len(next_inputs) == 0:
                next_inputs = prev_inputs
                prev_inputs = []
            # Collect the next batch.
            outputs = collections.defaultdict(list)
            for _ in range(self.batch_size):
                inputs = next_inputs.pop(0)
                for k, v in inputs.items():
                    outputs[k].extend(v)
                prev_inputs += self.worker_queue.get_n(1)
            # Stack batch data.
            if self.contiguous:
                if "latents" in outputs:
                    outputs["latents"] = np.stack(outputs["latents"])
            # Send batch data to consumer.
            self.batch_queue.put(outputs)


class FeatureDataLoader(DataLoader):
    """Loader to return the batch of data features."""

    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, FeatureWorker, **kwargs)
