#!/usr/bin/env python
# coding: utf-8
#
# -----------------------------------------------------------------------------
# Copyright (c) 2026 Martin Boldt, Blekinge Institute of Technology, Sweden.
#
# This file is part of the GraphVenn crime hotspot detection project.
#
# This work has been funded by the Swedish Research Council (grant 2022–05442).
#
# Licensed under the MIT License. You may obtain a copy of the License at:
# https://opensource.org/licenses/MIT
#
# This software is provided "as is", without warranty of any kind, express or
# implied, including but not limited to the warranties of merchantability,
# fitness for a particular purpose and noninfringement. In no event shall the
# authors or copyright holders be liable for any claim, damages or other
# liability, whether in an action of contract, tort or otherwise, arising
# from, out of or in connection with the software or the use or other dealings
# in the software.
# -----------------------------------------------------------------------------
#
import threading
import psutil

class PhaseMemTracker:
    def __init__(self, interval=0.2):
        self.interval = interval
        self._stop = threading.Event()
        self._peak_bytes = 0

    def _loop(self):
        proc = psutil.Process()
        while not self._stop.is_set():
            try:
                rss = proc.memory_info().rss
                if rss > self._peak_bytes:
                    self._peak_bytes = rss
            except Exception:
                pass
            self._stop.wait(self.interval)

    def start(self):
        self._stop.clear()
        self._peak_bytes = 0
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def stop(self):
        self._stop.set()
        self._t.join()

    @property
    def peak_mb(self):
        return self._peak_bytes / (1024.0 * 1024.0)