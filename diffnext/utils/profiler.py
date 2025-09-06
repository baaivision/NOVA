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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, esither express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""Profiler utilities."""

import collections
import contextlib
import datetime
import time
import numpy as np


class SmoothedValue(object):
    """Track values and provide smoothed report."""

    def __init__(self, window_size=None, fmt=None):
        self.fmt = fmt or "{median:.4f} ({mean:.4f})"
        self.deque = collections.deque(maxlen=window_size)

    def __str__(self):
        return self.fmt.format(value=self.value, mean=self.mean, median=self.median)

    @property
    def value(self):
        return self.deque[-1]

    @property
    def mean(self):
        return np.mean(self.deque)

    @property
    def median(self):
        return np.median(self.deque)

    def update(self, value):
        self.deque.append(value)


class Timer(object):
    """Simple timer."""

    def __init__(self):
        self.total_time = 0.0
        self.calls = 0
        self.start_time = 0.0
        self.diff = 0.0
        self.average_time = 0.0

    def add_diff(self, diff, n=1, average=False):
        self.total_time += diff
        self.calls += n
        self.average_time = self.total_time / self.calls
        return self.average_time if average else self.diff

    @contextlib.contextmanager
    def tic_and_toc(self, n=1):
        try:
            yield self.tic()
        finally:
            self.toc(n)

    def tic(self):
        self.start_time = time.time()
        return self

    def toc(self, n=1, average=False):
        self.diff = time.time() - self.start_time
        return self.add_diff(self.diff, n, average)


def get_progress(timer, step, max_steps):
    """Return the progress information."""
    eta_seconds = timer.average_time * (max_steps - step)
    eta = str(datetime.timedelta(seconds=int(eta_seconds)))
    progress = (step + 1.0) / max_steps
    return "< PROGRESS: {:.2%} | SPEED: {:.3f}s / step | ETA: {} >".format(
        progress, timer.average_time, eta
    )
