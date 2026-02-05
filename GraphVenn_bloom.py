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
import numpy as np

class TinyBloom:
    """
    Small Bloom signatures used only as a precheck for dominance.
    They never decide dominance; they only tell us when exact set-check
    is unnecessary (not a superset) => safe & fast.
    """
    __slots__ = ("bits", "m", "seeds")

    def __init__(self, m_bits=256, seeds=(0x9e3779b1, 0x7f4a7c15, 0x85ebca6b, 0xc2b2ae35)):
        self.m = int(m_bits)
        self.bits = np.zeros((self.m,), dtype=np.bool_)
        self.seeds = np.array(seeds, dtype=np.uint64)

    def add_indices(self, idx_arr: np.ndarray):
        """Vectorized add of many integer indices."""
        if idx_arr is None:
            return
        idx_arr = np.asarray(idx_arr)
        if idx_arr.size == 0:
            return
        x = idx_arr.astype(np.uint64, copy=False)

        for s in self.seeds:
            y = x ^ np.uint64(s)
            y ^= (y >> np.uint64(33))
            y *= np.uint64(0xff51afd7ed558ccd)
            y ^= (y >> np.uint64(33))
            y *= np.uint64(0xc4ceb9fe1a85ec53)
            y ^= (y >> np.uint64(33))
            h = (y % np.uint64(self.m)).astype(np.int64, copy=False)
            self.bits[h] = True

    def maybe_subset_of(self, other: "TinyBloom") -> bool:
        """If any of our set bits are not set in other, we are NOT a subset.
        Returns True only if subset is still possible (or mismatch -> unknown).
        """
        if not isinstance(other, TinyBloom) or other.m != self.m:
            return True  # can't disprove; force exact check later
        pos = np.nonzero(self.bits)[0]
        if pos.size == 0:
            return True
        return bool(other.bits[pos].all())