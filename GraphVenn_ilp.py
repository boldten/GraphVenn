#!/usr/bin/env python
# coding: utf-8
#
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
# ILP helper routines for GraphVenn (CBC via PuLP)
#
from __future__ import annotations

import time
import sys
import numpy as np
import pulp

def _cbc_cmd(**kwargs):
    """Create a CBC solver command compatible across PuLP versions."""
    try:
        return pulp.PULP_CBC_CMD(**kwargs, use_mps=False)
    except TypeError:
        return pulp.PULP_CBC_CMD(**kwargs)

def _solve_monolithic_indices(H, N, get_cov, weights, L, threads=1, print_ilp=True):
    """Exact ILP (weighted max coverage, no double counting). Returns indices (into H)."""
    if print_ilp:
        print("  + Building ILP for exact maximum coverage (no double counting):")

    x = [pulp.LpVariable(f"x_{i}", 0, 1, cat="Binary") for i in range(len(H))]
    y = [pulp.LpVariable(f"y_{j}", 0, 1, cat="Binary") for j in range(L)]

    # inverse map
    t0 = time.time(); sys.stdout.write("     - Building inverse map location→candidates: ... "); sys.stdout.flush()
    loc_to_cands = [[] for _ in range(L)]
    for i in range(len(H)):
        cov = get_cov(i)
        for j in cov:
            loc_to_cands[j].append(i)
    sys.stdout.write(f"Done ({time.time()-t0:.2f}s)\n")

    prob = pulp.LpProblem("MaxCoverageNoDoubleCountWeighted", pulp.LpMaximize)
    prob += pulp.lpSum(weights[j] * y[j] for j in range(L))

    t0 = time.time(); sys.stdout.write("     - Adding linking constraints: ... "); sys.stdout.flush()
    for j, cand_list in enumerate(loc_to_cands):
        if cand_list:
            prob += y[j] <= pulp.lpSum(x[i] for i in cand_list)
    sys.stdout.write(f"Done ({time.time()-t0:.2f}s)\n")

    prob += pulp.lpSum(x) <= N

    print(f"     - ILP size preview: #candidates={len(H):,}, #unique_locations={L:,}")

    t0 = time.time(); sys.stdout.write("     - Solving ILP (CBC): ... "); sys.stdout.flush()
    show_ilp_msg=False # force CBC to print if false
    solver = _cbc_cmd(
        msg=show_ilp_msg,          
        threads=threads,
        timeLimit=None,     # no cutoff, identical optimality
    )
    _ = prob.solve(solver)
    sys.stdout.write(f"Done ({time.time()-t0:.2f}s)\n")

    return [i for i, var in enumerate(x) if pulp.value(var) > 0.5]

def _solve_part_curve(cid, cand_idx, crime_idx, maxK, get_cov, weights, threads=1):
    """Exact f_r(k) curve for a component with weighted objective."""
    import pulp
    if len(cand_idx) == 0 or len(crime_idx) == 0:
        return [0]*(maxK+1), None

    local_index = {g:i for i, g in enumerate(cand_idx)}
    crime_pos   = {g:i for i, g in enumerate(crime_idx)}
    C = [[] for _ in range(len(crime_idx))]
    for gi in cand_idx:
        cov = get_cov(gi)
        for j in cov:
            if j in crime_pos:
                C[crime_pos[j]].append(local_index[gi])

    local_weights = np.array([weights[g] for g in crime_idx], dtype=int)

    x = [pulp.LpVariable(f"x_c{cid}_{i}", 0, 1, cat="Binary") for i in range(len(cand_idx))]
    y = [pulp.LpVariable(f"y_c{cid}_{j}", 0, 1, cat="Binary") for j in range(len(crime_idx))]
    prob = pulp.LpProblem(f"MaxCov_Component_{cid}", pulp.LpMaximize)

    for j, cand_list in enumerate(C):
        if cand_list:
            prob += y[j] <= pulp.lpSum(x[i] for i in cand_list)
    prob += pulp.lpSum(local_weights[j] * y[j] for j in range(len(crime_idx)))

    cap = min(maxK, len(cand_idx))
    f = [0]*(cap+1)
    budget_con = pulp.LpConstraint(e=pulp.lpSum(x), sense=pulp.LpConstraintLE, rhs=0, name=f"budget_c{cid}")
    prob += budget_con

    thr = 1 if len(cand_idx) < 2000 else threads

    for k in range(cap + 1):
        budget_con.changeRHS(k)

        # IMPORTANT: recreate solver each solve (prevents TMPDIR / memory blowup)
        solver = _cbc_cmd(
            msg=False,
            threads=thr
        )

        _ = prob.solve(solver)

        f[k] = int(
            round(
                sum(
                    local_weights[j] * pulp.value(y[j])
                    for j in range(len(crime_idx))
                )
            )
        )

    return f, None

def _solve_part_select_indices(cid, cand_idx, crime_idx, K, get_cov, weights, threads=1):
    """Recover chosen candidates at fixed K (component)."""
    import pulp
    if K <= 0 or len(cand_idx) == 0 or len(crime_idx) == 0:
        return []
    local_index = {g:i for i, g in enumerate(cand_idx)}
    crime_pos   = {g:i for i, g in enumerate(crime_idx)}
    C = [[] for _ in range(len(crime_idx))]
    for gi in cand_idx:
        cov = get_cov(gi)
        for j in cov:
            if j in crime_pos:
                C[crime_pos[j]].append(local_index[gi])

    local_weights = np.array([weights[g] for g in crime_idx], dtype=int)

    x = [pulp.LpVariable(f"x_pick_c{cid}_{i}", 0, 1, cat="Binary") for i in range(len(cand_idx))]
    y = [pulp.LpVariable(f"y_pick_c{cid}_{j}", 0, 1, cat="Binary") for j in range(len(crime_idx))]
    prob = pulp.LpProblem(f"Pick_Component_{cid}", pulp.LpMaximize)

    for j, cand_list in enumerate(C):
        if cand_list:
            prob += y[j] <= pulp.lpSum(x[i] for i in cand_list)
    prob += pulp.lpSum(x) <= K
    prob += pulp.lpSum(local_weights[j] * y[j] for j in range(len(crime_idx)))

    show_ilp_msg=False # force CBC to print if True
    solver = _cbc_cmd(
        msg=show_ilp_msg,          
        threads=threads,
        timeLimit=None,     # no cutoff, identical optimality
    )
    _ = prob.solve(solver)
    chosen_local = [i for i, var in enumerate(x) if pulp.value(var) > 0.5]
    return [cand_idx[i] for i in chosen_local]
