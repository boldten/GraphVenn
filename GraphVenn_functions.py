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
import pandas as pd
import geopandas as gpd
import math
import networkx as nx
from shapely.geometry import Point
from shapely.ops import transform
from math import cos, radians
from haversine import haversine
from scipy.spatial.distance import cdist
from sklearn.neighbors import BallTree
from collections import defaultdict
import heapq
import pyproj
from multiprocessing import cpu_count
import logging
import time
import sys
import os
import pickle
import tempfile
import shutil
import atexit
import signal
from pathlib import Path
import folium

from GraphVenn_ilp import _solve_monolithic_indices, _solve_part_curve, _solve_part_select_indices
from GraphVenn_mem import PhaseMemTracker
from GraphVenn_bloom import TinyBloom

# --------------------------------------------------
# Variable initialization
# --------------------------------------------------

EARTH_RADIUS = 6371000.0 # Used for conversion between degrees and meters
STR_WIDTH = 110 # Approximate string output width
INPUT_COLUMNS = {'crime_code', 'year', 'latitude', 'longitude'} # Required columns in input CSV file

results_path='Results/'
verbose = 1
str_width = 110 # Approximate string output width

logging.basicConfig( filename="progress.log", format='%(asctime)s %(message)s', filemode='a' )
logger = logging.getLogger()
logger.setLevel( logging.DEBUG )


# --------------------------------------------------
# Robust temp directory for CBC / PuLP
# --------------------------------------------------

# Create a private temp directory in the user's home
_base_tmp = Path.home() / ".graphvenn_tmp"
_base_tmp.mkdir(parents=True, exist_ok=True)
# Create a unique run-specific subdirectory
_RUN_TMPDIR = tempfile.mkdtemp(prefix="cbc_", dir=_base_tmp)
# Tell *everything* (CBC, PuLP, Python) to use it
os.environ["TMPDIR"] = _RUN_TMPDIR
os.environ["TEMP"]   = _RUN_TMPDIR
os.environ["TMP"]    = _RUN_TMPDIR

# --------------------------------------------------
# Cleanup logic (normal exit + crashes + Ctrl-C)
# --------------------------------------------------

def _cleanup_tmpdir():
    try:
        shutil.rmtree(_RUN_TMPDIR, ignore_errors=True)
    except Exception:
        pass

# Normal Python exit
atexit.register(_cleanup_tmpdir)

# Handle Ctrl-C, SIGTERM, kernel kill attempts
def _signal_handler(signum, frame):
    _cleanup_tmpdir()
    sys.exit(128 + signum)

for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
    signal.signal(sig, _signal_handler)

# --------------------------------------------------
#             F U N C T I O N S
# --------------------------------------------------

def show_disclaimer():
    print( """Created by: Martin Boldt, Blekinge Institute of Technology, Sweden, 2026.
          
DISCLAIMER:
  This software is provided "as is", without warranty of any kind, express or implied, including but not limited to
  the warranties of merchantability, fitness for a particular purpose and noninfringement. In no event shall the
  author(s) or affiliated institution(s) be liable for any claim, damages or other liability, whether in an action
  of contract, tort or otherwise, arising from, out of or in connection with the software or the use or other dealings
  in the software.

  This code is intended for research and educational purposes only. It is not intended for operational use, nor
  should it be used for decision-making in law enforcement or public safety contexts without independent validation
  and appropriate oversight.

  Use at your own risk.
    """ )

def check_csv_file( crime_data_path ):
    """
    Check so that the CSV file can be opened, has the required columns.
    """
    if not os.path.isfile(crime_data_path):
        print(f"\n[Error] File not found: {crime_data_path}")
        print("Please check the filename and try again.\n")
        sys.exit(1)
    try:
        crimes_df = pd.read_csv( crime_data_path )
        missing_columns = INPUT_COLUMNS - set(crimes_df.columns)
        if missing_columns:
            print(f"Error: Missing required column(s): {', '.join(missing_columns)}")
            sys.exit(1)
    except Exception as e:
        print(f"Error while reading file: {crime_data_path}. Error: {e}")
        print("Exiting.")
        sys.exit(1)

def print_verbose( asked_verbose_level, string, ending="\n", flushing=False ):
    if ( verbose >= asked_verbose_level ):
        print( string, end=ending, flush=flushing )
        return len(string)
        
def store_to_file(title, data):
    try:
        pikd = open(title, 'wb')
        pickle.dump(data, pikd)
        pikd.close()
    except:
        print( "Error: Could not write to file: {}. Quitting".format( title ))
        exit(1)

def to_unique_gdf(df):
    """
    Convert unique_position_crime_counts to a GeoDataFrame with WGS84 geometry.
    Handles these cases:
    A) index is shapely Points, column 'count'
    B) df already has a 'geometry' column
    C) lat/lon are the first two columns after reset_index()
    """

    # Case A: index are shapely Points
    idx0 = df.index[0]
    if hasattr(idx0, "x") and hasattr(idx0, "y"):  # looks like a shapely Point
        tmp = df.reset_index().rename(columns={"index": "geometry"})
        gdf = gpd.GeoDataFrame(tmp, geometry="geometry", crs="EPSG:4326")
        return gdf

    # Case B: already has geometry column
    if "geometry" in df.columns:
        gdf = gpd.GeoDataFrame(df.copy(), geometry="geometry")
        if gdf.crs is None:
            gdf.set_crs(epsg=4326, inplace=True, allow_override=True)
        return gdf

    # Case C: lat/lon in first two columns after reset_index()
    tmp = df.reset_index()
    # ensure we have at least two columns to interpret as lat/lon
    if tmp.shape[1] >= 2:
        tmp = tmp.rename(columns={tmp.columns[0]: "latitude", tmp.columns[1]: "longitude"})
        if {"latitude", "longitude"}.issubset(tmp.columns):
            tmp["geometry"] = tmp.apply(lambda r: Point(float(r["longitude"]), float(r["latitude"])), axis=1)
            gdf = gpd.GeoDataFrame(tmp, geometry="geometry", crs="EPSG:4326")
            return gdf

    raise ValueError("Could not infer geometry for unique_position_crime_counts.")

def preprocess_data( df, show_output = True ):
    if show_output:
        print_verbose( 1, "* Preprocess data ... ", "" )

    df.index.name = 'id'        # Set the index name
    df.reset_index(inplace=True) # set up as index

    df = df.dropna(axis=0,how='any',subset=['latitude','longitude'])
    df['latitude'] = pd.to_numeric(df['latitude'])
    df['longitude'] = pd.to_numeric(df['longitude'])
    try:
        df = gpd.GeoDataFrame( df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs=4326 )
        df = df[list(INPUT_COLUMNS) + ['geometry']]
    except Exception as e:
        print(f"The .csv file must be comma (,) separated and contain these columns: {INPUT_COLUMNS}. Coordinates should be floats, e.g., 55.123 (note the dot, not comma).\n")
        print("The error was:", e)

    if show_output:
        print_verbose( 1, " done." )
        
    return df

def create_geodataframe( csv_path, show_output = True ):
    try:
        crimes_df = pd.read_csv( csv_path )
        crimes_df = preprocess_data( crimes_df, show_output )
    except Exception as e:
        print(f"An error occurred when reading the crime data from: {csv_path}.")
        print(f"The .csv file must be comma (,) separated and contain these columns (no capital letters):\n{INPUT_COLUMNS}. Lon/lat coordinates must be floats with, e.g., 55.123 (note the dot, not comma).\n")
        print("The error was:", e)

    unique_position_crime_counts = pd.DataFrame( crimes_df['geometry'].value_counts(sort=False) )
    unique_position_crime_counts.columns = ['count']
    
    return unique_position_crime_counts, crimes_df

def grid_haversine_distance(coord1, coord2):
    """Haversine formula to calculate the great circle distance in meters between two points."""
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1 
    dlon = lon2 - lon1 
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a)) 
    distance = 6371000 * c
    return distance

def calculate_area_gdf(gdf):
    lats = gdf.geometry.y
    lons = gdf.geometry.x
    top_right = (max(lats), max(lons))
    bottom_left = (min(lats), min(lons))
    width = grid_haversine_distance((bottom_left[0], top_right[1]), top_right)
    height = grid_haversine_distance(bottom_left, (bottom_left[0], top_right[1]))
    A = width * height
    A = A/1000000   # Return area in km^2    
    return A

def calculate_camera_area( radius_m ):
    a = math.pi * radius_m**2
    a = a/1000000   # Return area in km^2
    return a

# --- memory helpers.py (put near your other helpers) ---
def _bytes_to_mb(x): 
    return float(x) / (1024.0 * 1024.0)

def current_mem_mb():
    """Current RSS in MB."""
    try:
        import psutil
        p = psutil.Process()
        return _bytes_to_mb(p.memory_info().rss)
    except Exception:
        # Fallback: best-effort with resource (Unix)
        try:
            import resource
            # ru_maxrss is *current* RSS on some Unixes, but often peak; prefer psutil when available.
            return _bytes_to_mb(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
                                if sys.platform.startswith("darwin") else
                                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)  # normalize to bytes
        except Exception:
            return float('nan')


# ========================= GraphVenn (fast + globally optimal) =========================
#  Entry point:
#   run_graphvenn_method(...)
# ======================================================================================

def _eta_str(t0, done, total):
    elapsed = time.time() - t0
    if done <= 0:
        return "ETA: --"
    rate = elapsed / max(done, 1)
    remaining = max(total - done, 0) * rate
    h, rem = divmod(int(remaining), 3600)
    m, s = divmod(rem, 60)
    return f"ETA: {h}h{m}m{s}s"

def utm_epsg_for_lonlat(lon, lat):
    zone = int((lon + 180) // 6) + 1
    return f"EPSG:{32600 + zone if lat >= 0 else 32700 + zone}"

def _make_bloom_from_cov(cov_idx, m_bits=256):
    b = TinyBloom(m_bits=m_bits)
    # cov_idx is already a NumPy array of ints; add in one shot
    b.add_indices(cov_idx)
    return b

# ------------------ assign unique counts with priority ----------------
def assign_unique_counts_with_priority(selected_idx, H, L, get_cov, weights, priority="standalone_desc"):
    """
    selected_idx: list of candidate indices (into H)
    H: list of (standalone_count, (lat,lon))
    L: number of unique locations (0..L-1)
    get_cov: i -> np.ndarray of covered location indices (subset of 0..L-1)
    weights: per-location incident counts
    priority:
       'standalone_desc' -> larger standalone first
       list/array        -> explicit order (earlier = higher priority)
    Returns:
       assigned_list_sorted: [(unique_count, (lat,lon), idx), ...] sorted by unique_count desc
       assigned_list_order : in the priority order (for logging)
    """
    if isinstance(priority, (list, tuple, np.ndarray)):
        order = list(priority)
    elif priority == "standalone_desc":
        order = sorted(selected_idx, key=lambda i: H[i][0], reverse=True)
    else:
        order = list(selected_idx)

    assigned = np.zeros(L, dtype=bool)
    per_hotspot_counts = {}
    for i in order:
        cov_i = get_cov(i)
        new = cov_i[~assigned[cov_i]]
        per_hotspot_counts[i] = int(weights[new].sum())
        assigned[new] = True

    assigned_list_order  = [(per_hotspot_counts[i], H[i][1], i) for i in order]
    assigned_list_sorted = sorted(assigned_list_order, key=lambda t: t[0], reverse=True)
    return assigned_list_sorted, assigned_list_order

def _print_optimal_summary(optimal_hotspots, standalone_counts_for_selected):
    total_unique = sum(c for (c, _) in optimal_hotspots)
    k = min(10, len(optimal_hotspots))
    total_standalone = int(sum(standalone_counts_for_selected))
    overlap = total_standalone - total_unique
    print(f"  + Globally optimal top-{len(optimal_hotspots)} hotspots detected.")
    print(f"    - Total unique crimes covered (Optimal): {total_unique:,}")
    print(f"    - Top-{k} Optimal (sorted by unique count): {[c for (c, _) in optimal_hotspots[:k]]}")
    print(f"    - Diagnostics: sum(standalone over selected)={total_standalone:,}  overlap={overlap:,}")

def OptimizeHotspots(H, N, d, all_crimes_gdf, min_hotspot_size, strategy='greedy', graph_components=None, graph_loc_keys=None, timings=None, print_ilp=True, mem_hooks=None):
    """
    H: [(standalone_count, (lat,lon)), ...] candidates (count-only)
    N: number to select
    d: radius (meters)
    all_crimes_gdf: incidents (Point) in any CRS (will be converted)
    min_hotspot_size: minimum crime incidents that must exist to form a hotspot (default 1)
    strategy: search strategy (default 'greedy')
    ...
    Returns (optimal_hotspots, greedy_hotspots) each sorted by unique count desc.
    """
    valid = {"greedy","optimal","both"}

    if timings is None:
        timings = {"descriptive": 0.0, "greedy": 0.0, "optimal": 0.0}
    if mem_hooks is None:
        mem_hooks = {}
    start_mem = mem_hooks.get("start", lambda phase: None)
    stop_mem  = mem_hooks.get("stop",  lambda phase: None)
        
    if strategy not in valid:
        raise ValueError(f"Invalid strategy={strategy}. Choose from {sorted(valid)}")

    num_threads = 1  # always use 1 thread during experiments
    #if num_threads is None:
    #    try:
    #        num_threads = max(1, (os.cpu_count() or 1) - 1)
    #    except Exception:
    #        num_threads = 1

    print(f"  + Searching for top-{N} suitable hotspots among detected locations (threads={num_threads}).")

    # Project candidate centers to local CRS
    t0 = time.time(); sys.stdout.write("  + Picking projected CRS and projecting hotspot centers: ... "); sys.stdout.flush()
    centers_lon = np.array([lon for _, (_, lon), *_ in H], dtype=float)
    centers_lat = np.array([lat for _, (lat, _), *_ in H], dtype=float)
    center_lon = float(np.mean(centers_lon)); center_lat = float(np.mean(centers_lat))
    crs_proj = utm_epsg_for_lonlat(center_lon, center_lat)
    to_proj  = pyproj.Transformer.from_crs("EPSG:4326", crs_proj, always_xy=True)
    cx, cy   = to_proj.transform(centers_lon, centers_lat)
    centers_xy = np.column_stack([cx, cy]).astype(np.float64)
    sys.stdout.write(f"Done ({time.time()-t0:.2f}s)\n")

    # Collapse incidents -> unique locations (weights), build BallTree in same CRS
    t0 = time.time(); sys.stdout.write("  + Projecting unique crime locations & building BallTree: ... "); sys.stdout.flush()
    rep = all_crimes_gdf if (all_crimes_gdf.crs and all_crimes_gdf.crs.to_string()=="EPSG:4326") \
                          else all_crimes_gdf.to_crs(epsg=4326)
    uniq = (rep.groupby(rep.geometry)
              .size()
              .reset_index(name="count")
              .set_geometry("geometry"))
    lon = uniq.geometry.x.to_numpy(float); lat = uniq.geometry.y.to_numpy(float)
    qx, qy = to_proj.transform(lon, lat)
    loc_xy = np.column_stack([qx, qy]).astype(np.float64)   # Lx2
    weights = uniq["count"].to_numpy(int)                   # L
    tree = BallTree(loc_xy, metric="euclidean")
    L = loc_xy.shape[0]
    sys.stdout.write(f"Done ({time.time()-t0:.2f}s)\n")

    # Coverage cache (candidate i -> np.ndarray[int] of covered locations)
    coverage_cache = {}
    def get_cov(i):
        c = coverage_cache.get(i)
        if c is None:
            c = tree.query_radius([centers_xy[i]], r=d, count_only=False)[0].astype(np.int32, copy=False)
            coverage_cache[i] = c
        return c

    # ----------------------- GREEDY -----------------------
    greedy_hotspots = []
    greedy_sel_idx = []
    if strategy in {"greedy", "both"}:
        t0 = time.time(); sys.stdout.write("  + Greedy selection (no double counting): ... "); sys.stdout.flush()
        t_g = time.time()
        start_mem("greedy")

        assigned = np.zeros(L, dtype=bool)

        # initial marginal = standalone (upper bound)
        standalone = np.array([c for c, _ in H], dtype=int)
        heap = [(-int(standalone[i]), 0, i) for i in range(len(H)) if standalone[i] > 0]
        heapq.heapify(heap)
        version = np.zeros(len(H), dtype=int)

        every = max(1, max(1, len(H)//100))
        pops = 0
        while len(greedy_sel_idx) < N and heap:
            neg_gain_guess, ver, i = heapq.heappop(heap)
            pops += 1
            if ver != version[i]:
                continue  # stale

            # recompute true marginal gain for i wrt current 'assigned'
            cov = get_cov(i)
            if cov.size == 0:
                continue
            new_idx = cov[~assigned[cov]]
            gain = int(weights[new_idx].sum())

            # lazy update: if stale (guess != true), push back with updated key
            if -neg_gain_guess != gain:
                version[i] += 1
                heapq.heappush(heap, (-gain, version[i], i))
                continue

            # accept this candidate
            if gain <= 0:
                continue
            greedy_sel_idx.append(i)
            assigned[new_idx] = True

            # progress print (optional)
            if pops % every == 0 or len(greedy_sel_idx) == N:
                sys.stdout.write(
                    f"\r     - Considered {pops:,}/{len(H):,} pops   selected={len(greedy_sel_idx):,}   {_eta_str(t0, pops, len(H))}"
                ); sys.stdout.flush()

        sys.stdout.write("\n")

        # Assign uniques using the actual greedy pick order; then sort for reporting
        greedy_sorted, _ = assign_unique_counts_with_priority(
            greedy_sel_idx, H, L, get_cov, weights, priority=greedy_sel_idx
        )
        greedy_hotspots = [(c, ll) for (c, ll, _) in greedy_sorted]
        greedy_unique_total = sum(c for (c, _) in greedy_hotspots)
        k = min(10, len(greedy_hotspots))
        print(f"  + Selected {len(greedy_hotspots)} hotspots (strategy='greedy').")
        print(f"    - Total unique crimes covered (Greedy): {greedy_unique_total:,}")
        print(f"    - Top-{k} Greedy (sorted by unique count): {[c for (c, _) in greedy_hotspots[:k]]}")
        standalone_sum = int(sum(H[i][0] for i in greedy_sel_idx))
        overlap = standalone_sum - greedy_unique_total
        print(f"    - Diagnostics (Greedy): sum(standalone)={standalone_sum:,}  overlap={overlap:,}")

        stop_mem("greedy")
        timings["greedy"] = timings.get("greedy", 0.0) + (time.time() - t_g)

    if strategy == "greedy":
        return [], greedy_hotspots

    # ----------------- Continue with Global Optimal through ILP -----------------
    # Hybrid: Greedy floor, exact dominance (guarded by Bloom), and 2d-UB pruning.

    t_o = time.time()
    start_mem("optimal")

    # 2d-UB for a candidate i: sum of weights within 2d of its center.
    # Precompute a helper that caches UB radii queries.
    ub_cache = {}
    def ub_2d_for_idx(i):
        u = ub_cache.get(i)
        if u is not None: return u
        idxs = tree.query_radius([centers_xy[i]], r=2.0*d, count_only=False)[0]
        val = int(weights[idxs].sum())
        ub_cache[i] = val
        return val

    # Floor = greedy total (unique) or 0 if no greedy
    floor = sum(c for (c, _) in greedy_hotspots) if greedy_hotspots else 0

    # Dedupe snapped centers and keep a small set of "keepers" per neighborhood
    # Use a neighborhood key ~ snap to 0.5*d in meters on XY
    cell = max(d*0.5, 1.0)
    nb_key = np.floor(centers_xy / cell).astype(np.int64)
    buckets = defaultdict(list)
    for i, key in enumerate(map(tuple, nb_key)):
        buckets[key].append(i)

    # Optionally cap keepers per bucket for throughput; exact dominance still safe.
    KEEPER_CAP = 16  # D1: tweak as needed

    # For each bucket, run UB prune + dominance prune relative to a small keeper set.
    # Keepers store: {idx, count, cov (optional), bloom}
    pruned_mask = np.zeros(len(H), dtype=bool)

    def exact_dominates(covA, cntA, covB, cntB):
        # B dominates A if covA ⊆ covB and cntB >= cntA
        if cntB < cntA: return False
        # exact subset: all elements of A are in B
        # Use np.in1d on sorted arrays for speed
        if covA.size == 0: return True
        if covB.size == 0: return False
        a = np.sort(covA); b = np.sort(covB)
        return np.in1d(a, b, assume_unique=False).all()


    # prune only if even the 2d upper bound is below the minimum hotspot size
    MIN_ACCEPT = max(1, min_hotspot_size)

    for key, idx_list in buckets.items():
        # Pre-order by standalone desc to give dominance a chance early.
        idx_list.sort(key=lambda i: H[i][0], reverse=True)
        keepers = []  # list of dicts
        for i in idx_list:
            if pruned_mask[i]: continue

            # Light UB prune
            if ub_2d_for_idx(i) < MIN_ACCEPT:
                pruned_mask[i] = True
                continue

            # Build / fetch coverage once (we need exact dominance)
            cov_i = get_cov(i)
            cnt_i = int(weights[cov_i].sum())

            # Bloom precheck against keepers for possible domination both directions.
            # - If candidate is dominated by any keeper -> prune i.
            # - If candidate dominates some keepers -> drop those keepers.
            dominated = False
            bloom_i = _make_bloom_from_cov(cov_i)

            to_remove = []
            for kpos, K in enumerate(keepers):
                # quick reject using bloom (if i ⊆ K ?)
                if bloom_i.maybe_subset_of(K["bloom"]):
                    if exact_dominates(cov_i, cnt_i, K["cov"], K["cnt"]):
                        dominated = True
                        break
                # check other direction (K ⊆ i ?) to possibly replace keepers
                if K["bloom"].maybe_subset_of(bloom_i):
                    if exact_dominates(K["cov"], K["cnt"], cov_i, cnt_i):
                        to_remove.append(kpos)

            if dominated:
                pruned_mask[i] = True
                continue

            # Remove dominated keepers
            if to_remove:
                for pos in reversed(sorted(to_remove)):
                    keepers.pop(pos)

            # Admit i as keeper (cap size)
            keepers.append({"idx": i, "cnt": cnt_i, "cov": cov_i, "bloom": bloom_i})
            if len(keepers) > KEEPER_CAP:
                # keep the best by cnt (stable)
                keepers.sort(key=lambda d: d["cnt"], reverse=True)
                keepers = keepers[:KEEPER_CAP]

        # Mark survivors in this bucket as "not pruned"; others already marked
        # (nothing else to do here)

    # Build pruned H'
    survivors = [i for i in range(len(H)) if not pruned_mask[i]]
    if len(survivors) < len(H):
        print(f"  + Pruning reduced candidates: {len(H):,} → {len(survivors):,} (kept {len(survivors)/len(H):.1%})")
    H_idx_map = {old: k for k, old in enumerate(survivors)}
    H2 = [H[i] for i in survivors]

    # coverage cache already populated for many; rebuild a thin wrapper for H2 indices
    def get_cov2(k):
        return get_cov(survivors[k])

    centers_xy2 = centers_xy[survivors]

    # ------------------- Component DP (C3) + fallback -------------------
    t0 = time.time(); sys.stdout.write("  + Decomposing by 2d location-connectivity: ... "); sys.stdout.flush()
    nbrs = BallTree(loc_xy, metric="euclidean").query_radius(loc_xy, r=2.0*d, count_only=False)
    parent = list(range(L))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    def union(a,b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[rb] = ra
    for j, neigh in enumerate(nbrs):
        # only union j<k to avoid double-work
        neigh = neigh[neigh>j]
        for k in neigh:
            union(j, int(k))
    comp_crimes = defaultdict(list)
    for j in range(L):
        comp_crimes[find(j)].append(j)
    components = [np.array(v, dtype=np.int32) for v in comp_crimes.values()]
    crime_to_comp = np.full(L, -1, dtype=np.int32)
    for cid, arr in enumerate(components):
        crime_to_comp[arr] = cid

    # map reduced-candidates to components
    comp_to_cands = defaultdict(list)
    cross_component = False
    for ki in range(len(H2)):
        cov = get_cov2(ki)
        if cov.size == 0: continue
        comps = np.unique(crime_to_comp[cov]); comps = comps[comps>=0]
        if comps.size == 0: continue
        if comps.size > 1:
            cross_component = True
            break
        comp_to_cands[int(comps[0])].append(ki)

    if cross_component or len(components) == 1:
        sys.stdout.write(f"Done ({time.time()-t0:.2f}s) — no valid split (monolithic ILP).\n")
        idxs2 = _solve_monolithic_indices(H2, N, get_cov2, weights, L, threads=num_threads, print_ilp=print_ilp)
        optimal_sorted, _ = assign_unique_counts_with_priority(idxs2, H2, L, get_cov2, weights, priority="standalone_desc")
        optimal_hotspots = [(c, ll) for (c, ll, _) in optimal_sorted]
        _print_optimal_summary(optimal_hotspots, [H2[i][0] for i in idxs2])
        return optimal_hotspots, greedy_hotspots

    sys.stdout.write(f"Done ({time.time()-t0:.2f}s) — found {len(components)} components.\n")

    # Per-component curves and DP
    def solve_part_curve(cid, cand_idx, crime_idx, maxK):
        return _solve_part_curve(cid, cand_idx, crime_idx, maxK, get_cov2, weights, threads=num_threads)

    comp_order = sorted([cid for cid in comp_to_cands if len(comp_to_cands[cid])>0],
                        key=lambda c: len(comp_to_cands[c]), reverse=True)
    if not comp_order:
        idxs2 = _solve_monolithic_indices(H2, N, get_cov2, weights, L, threads=num_threads, print_ilp=print_ilp)
        optimal_sorted, _ = assign_unique_counts_with_priority(idxs2, H2, L, get_cov2, weights, priority="standalone_desc")
        optimal_hotspots = [(c, ll) for (c, ll, _) in optimal_sorted]
        _print_optimal_summary(optimal_hotspots, [H2[i][0] for i in idxs2])
        return optimal_hotspots, greedy_hotspots

    sys.stdout.write("  + Analyzing components: "); sys.stdout.flush()
    t_start = time.time()
    f_vectors = {}
    for cid in comp_order:
        f_vectors[cid], _ = solve_part_curve(cid, comp_to_cands[cid], components[cid].tolist(), N)
    sys.stdout.write(f"Done ({time.time()-t_start:.2f}s)\n")

    # DP allocate N
    sys.stdout.write("  + Allocating budget across components (DP): ... "); sys.stdout.flush()

    # Per-component capacity (cannot pick more than len(fr)-1 from a component)
    cap_per_comp = {cid: min(N, len(f_vectors[cid]) - 1) for cid in comp_order}
    total_cap    = sum(cap_per_comp.values())
    N_eff        = min(N, total_cap)  # clamp to available capacity

    best   = [0] * (N_eff + 1)
    choice = [[0] * (N_eff + 1) for _ in range(len(comp_order))]

    for r, cid in enumerate(comp_order):
        fr  = f_vectors[cid]               # fr[k] = best coverage using k hotspots in this component
        cap = cap_per_comp[cid]
        new = best[:]                      # previous row -> new row
        for b in range(N_eff + 1):
            ub = min(b, cap)
            for k in range(ub + 1):
                val = best[b - k] + fr[k]
                # prefer larger k on ties to help consume the budget
                if (val > new[b]) or (val == new[b] and k > choice[r][b]):
                    new[b]      = val
                    choice[r][b] = k
        best = new
    sys.stdout.write("Done\n")

    # ---- Backtrack allocation (consume the whole N_eff if possible) ----
    alloc = {cid: 0 for cid in comp_order}
    b = N_eff
    for r in range(len(comp_order) - 1, -1, -1):
        k = choice[r][b]
        cid = comp_order[r]
        alloc[cid] = k
        b -= k

    # Safety valve: if leftover (b > 0) because of tie pathology, pack remaining into any component with capacity
    if b > 0:
        for cid in comp_order:
            cap = cap_per_comp[cid]
            add = min(b, cap - alloc[cid])
            if add > 0:
                alloc[cid] += add
                b -= add
                if b == 0:
                    break
    # (optional) assert b == 0, "DP backtrack did not allocate full budget"

    # You can report how many will be picked in total:
    picked_total = sum(alloc.values())

    # solve each component at allocated K
    all_selected = []
    if print_ilp:
        sys.stdout.write("  + Solving components at allocated budgets (exact): "); sys.stdout.flush()
        t_start = time.time()

    for cid in comp_order:
        k = alloc[cid]
        if print_ilp and k > 0:
            sys.stdout.write("."); sys.stdout.flush()
        if k == 0: 
            continue
        chosen = _solve_part_select_indices(
            cid, comp_to_cands[cid], components[cid].tolist(),
            k, get_cov2, weights, threads=num_threads
        )
        all_selected.extend(chosen)

    if print_ilp:
        sys.stdout.write(f" Done ({time.time()-t_start:.2f}s)\n")

    optimal_sorted, _ = assign_unique_counts_with_priority(all_selected, H2, L, get_cov2, weights, priority="standalone_desc")
    optimal_hotspots = [(c, ll) for (c, ll, _) in optimal_sorted]
    _print_optimal_summary(optimal_hotspots, [H2[i][0] for i in all_selected])

    stop_mem("optimal")
    timings["optimal"] = timings.get("optimal", 0.0) + (time.time() - t_o)

    return optimal_hotspots, greedy_hotspots

# --------------------------- GraphVenn front-end ---------------------------

def run_graphvenn_method(unique_position_crime_counts, current_max_camera_coverage, spatial_resolution, N, all_crimes_gdf, min_hotspot_size=1, strategy="both"):
    """
    GraphVenn front-end: generates candidate hotspot centers by snapped p-grid
    around each unique crime location, caches only counts per grid cell,
    keeps a running top-N of disjoint centers for pruning, and delegates
    final selection to OptimizeHotspots().

    Returns
    -------
    optimal_hotspots, greedy_hotspots, timings
        Each hotspot as (count, (lat, lon)), and timings contain the execution time and RAM memory usage
    """
    print(f"x Running GraphVenn(N={N}, d={current_max_camera_coverage}, p={spatial_resolution}, strategy='{strategy}')")

    # ---- Initialize phase runtime timers ----
    timings = {"descriptive": 0.0, "greedy": 0.0, "optimal": 0.0}
    t_phase = time.time()
    mem = PhaseMemTracker(); mem.start()
    two_d = 2.0 * current_max_camera_coverage

    # ---------- Normalize input GDF to EPSG:4326 ----------
    if not isinstance(unique_position_crime_counts, gpd.GeoDataFrame):
        df = unique_position_crime_counts.reset_index()
        geom_col = None
        for col in df.columns:
            try:
                if df[col].apply(lambda v: isinstance(v, Point)).all():
                    geom_col = col
                    break
            except Exception:
                pass
        if geom_col is None:
            raise ValueError("Error: Could not find a Point geometry column in DataFrame.")
        unique_position_crime_counts = gpd.GeoDataFrame(df, geometry=geom_col, crs="EPSG:4326")
    if unique_position_crime_counts.crs is None:
        unique_position_crime_counts.set_crs(epsg=4326, inplace=True)
    elif unique_position_crime_counts.crs.to_string() != "EPSG:4326":
        unique_position_crime_counts = unique_position_crime_counts.to_crs(epsg=4326)

    # ---------- Pick local metric CRS & project unique positions ----------
    print("  + Preparing data and building BallTree: ... ", end="")
    rep_wgs84 = unique_position_crime_counts
    center_lat = float(rep_wgs84.geometry.y.mean())
    center_lon = float(rep_wgs84.geometry.x.mean())
    crs_proj = utm_epsg_for_lonlat(center_lon, center_lat)

    crimes_proj = unique_position_crime_counts.to_crs(crs_proj)
    coords = np.array([(pt.x, pt.y) for pt in crimes_proj.geometry.values], dtype=float)  # (M,2)
    weights = np.array([row.count for row in crimes_proj.itertuples()], dtype=np.int32)

    # BallTree over unique positions (meters)
    tree = BallTree(coords, metric="euclidean")

    # Transformer lon/lat -> metric CRS (vectorized) for candidate projection
    to_proj = pyproj.Transformer.from_crs("EPSG:4326", crs_proj, always_xy=True)
    print("Done.")

    G = nx.Graph()
    M = coords.shape[0]

    # Add nodes: one per unique crime location
    # Node id = index into coords / weights arrays
    for i in range(M):
        G.add_node(i, weight=int(weights[i]))

    # Add edges for locations within distance d
    # IMPORTANT: reuse the SAME BallTree (no extra distance work)
    for i in range(M):
        idxs = tree.query_radius(coords[i].reshape(1, -1), r=two_d)[0]
        for j in idxs:
            if j > i:   # already implies j != i
                G.add_edge(i, int(j))

    # Extract connected components (Phase-1 output)
    graph_components = [list(c) for c in nx.connected_components(G)]

    if len(graph_components) > 1:
        tot = sum(len(c) for c in graph_components)
        #print(f"  + Phase-1 graph components: {len(graph_components):,} (total nodes={tot:,})")
        sizes = np.array([len(c) for c in graph_components], dtype=np.int32)
        print(
            f"  + Created graph with {len(sizes):,} components (total nodes: {sizes.sum():,}) | "
            f"min={sizes.min()}, p50={int(np.median(sizes))}, p90={int(np.quantile(sizes,0.9))}, "
            f"p99={int(np.quantile(sizes,0.99))}, max={sizes.max()}"
        )

    # ---------- p-grid step on lon/lat ----------
    p = int(spatial_resolution)
    step = 10.0 ** (-p)  # strict p-decimal grid step

    # ---------- Working sets (persist across seeds) ----------
    evaluated_cells = {}   # (lat_p, lon_p) -> count
    added_keys      = set()  # (lat_p, lon_p) we've already appended to H (dedupe)

    H = []                 # candidates: (standalone_count, (lat, lon)) in EPSG:4326
    selected = []          # running best-so-far disjoint list [(cnt,(lat,lon)), ...], desc by cnt
    selected_xy = []       # corresponding (x,y) in meters for disjoint checks

    def is_disjoint(xy_m):
        if not selected_xy:
            return True
        dx = np.array([xy_m[0] - sx for sx, _ in selected_xy], float)
        dy = np.array([xy_m[1] - sy for _, sy in selected_xy], float)
        return np.all(np.hypot(dx, dy) >= two_d)

    def nth_score():
        if len(selected) < N:
            return float("-inf")
        return selected[-1][0]

    def try_insert_disjoint(cnt, lat_r, lon_r, xy_m):
        """Insert (cnt,(lat,lon)) into selected lists if disjoint; keep top-N by cnt."""
        if not is_disjoint(xy_m):
            return
        entry = (cnt, (lat_r, lon_r))
        # keep 'selected' sorted descending by cnt (binary insertion by negative key)
        pos = np.searchsorted([-e[0] for e in selected], -cnt, side="left")
        selected.insert(pos, entry)
        selected_xy.insert(pos, xy_m)
        if len(selected) > N:
            selected.pop()
            selected_xy.pop()

    # ---------- Seeds (unique crime positions in both XY and lat/lon) ----------
    seeds_xy = coords
    crimes_wgs = crimes_proj.to_crs(epsg=4326)
    seeds_lat = crimes_wgs.geometry.y.values
    seeds_lon = crimes_wgs.geometry.x.values

    graph_loc_keys = [
        (round(float(lat), 7), round(float(lon), 7))
        for lat, lon in zip(seeds_lat, seeds_lon)
    ]

    # Upper bound around a seed within radius 2d
    def seed_upper_bound_2d(xy_m):
        idxs = tree.query_radius([xy_m], r=2.0 * current_max_camera_coverage, count_only=False)[0]
        return int(weights[idxs].sum())

    num_nodes = len(seeds_xy)
    every_Nth = max(1, num_nodes // 200)
    sys.stdout.write(f"\r  + Processed unique crime locations: 0 of {num_nodes:,} (Have N disjoint:  no, Pruning: off): ...")
    sys.stdout.flush()

    for it in range(num_nodes):
        xy_m     = seeds_xy[it]
        lat_seed = float(seeds_lat[it])
        lon_seed = float(seeds_lon[it])

        # progress line
        if (it + 1) % every_Nth == 0:
            if len(selected) >= N:
                sys.stdout.write(
                    f"\r  + Processed unique crime locations: {it+1:,} of {num_nodes:,} (Have N disjoint: yes, Pruning: activated): ...                   "
                ); sys.stdout.flush()
            else:
                sys.stdout.write(
                    f"\r  + Processed unique crime locations: {it+1:,} of {num_nodes:,} (Have N disjoint:  no, Pruning: off): ...                   "
                ); sys.stdout.flush()

        # pruning by 2d upper bound once we have N
        if len(selected) >= N:
            ub = seed_upper_bound_2d(xy_m)
            active_floor = max(min_hotspot_size, nth_score())
            if ub < active_floor:
                continue

        # bounding box in degrees around seed enclosing the radius-d disc
        dlat_deg = current_max_camera_coverage / 111_320.0
        coslat   = max(np.cos(np.radians(lat_seed)), 1e-6)
        dlon_deg = current_max_camera_coverage / (111_320.0 * coslat)

        lat_vals = np.arange(
            np.floor((lat_seed - dlat_deg) / step) * step,
            np.ceil((lat_seed + dlat_deg) / step) * step + step / 2.0,
            step
        )
        lon_vals = np.arange(
            np.floor((lon_seed - dlon_deg) / step) * step,
            np.ceil((lon_seed + dlon_deg) / step) * step + step / 2.0,
            step
        )

        # snap grid and project (lon, lat) order for transformer
        lat_grid, lon_grid = np.meshgrid(lat_vals, lon_vals, indexing="ij")
        lon_flat, lat_flat = lon_grid.ravel(), lat_grid.ravel()
        xg, yg = to_proj.transform(lon_flat, lat_flat)

        # mask to the radius-d circle around the seed in metric space
        dx = xg - xy_m[0]
        dy = yg - xy_m[1]
        mask = (dx * dx + dy * dy) <= (current_max_camera_coverage ** 2)

        # iterate over all snapped grid points inside the circle
        for LAT, LON, cx, cy in zip(lat_flat[mask], lon_flat[mask], xg[mask], yg[mask]):
            key = (round(float(LAT), p), round(float(LON), p))

            # skip duplicate snapped centers across ALL seeds
            if key in added_keys:
                continue

            # reuse cached count if seen, otherwise compute via BallTree
            if key in evaluated_cells:
                cnt = evaluated_cells[key]
            else:
                idxs = tree.query_radius([[cx, cy]], r=current_max_camera_coverage, count_only=False)[0]
                cnt = int(weights[idxs].sum())
                evaluated_cells[key] = cnt

            if cnt < min_hotspot_size:
                continue

            # record candidate (standalone count, (lat,lon)) and dedupe mark
            H.append((cnt, (key[0], key[1])))
            added_keys.add(key)

            # maintain running disjoint top-N (for early pruning)
            if (len(selected) < N) or (cnt > nth_score()):
                try_insert_disjoint(cnt, key[0], key[1], (cx, cy))

    # tidy progress line
    if len(selected) >= N:
        sys.stdout.write(
            f"\r  + Processed unique crime locations: {num_nodes:,} of {num_nodes:,} (Have N disjoint: yes, Pruning: activated): ... Done.                  \n"
        ); sys.stdout.flush()
    else:
        sys.stdout.write(
            f"\r  + Processed unique crime locations: {num_nodes:,} of {num_nodes:,} (Have N disjoint:  no, Pruning: off): ... Done.                  \n"
        ); sys.stdout.flush()

    print(f"  + Evaluated {len(evaluated_cells):,} unique grid cells.")
    print(f"  + Compiled candidate locations, which includes {min(len(selected), N)} disjoint hotspot candidates (≥ 2d apart).")

    # No pruning done, so just keep everything
    H_pruned = H  

    # stop descriptive timer and recoded the runtime & RAM memory usage
    timings["descriptive"] = time.time() - t_phase
    mem.stop()
    timings["descriptive_ram_peak_mb"] = mem.peak_mb

    # hooks for greedy/optimal phases
    _trackers = {}
    def start_mem(phase):
        t = PhaseMemTracker()
        t.start()
        _trackers[phase] = t

    def stop_mem(phase):
        t = _trackers.pop(phase, None)
        if t:
            t.stop()
            timings[f"{phase}_ram_peak_mb"] = t.peak_mb

    optimal_hotspots, greedy_hotspots = OptimizeHotspots(
        H_pruned, N, current_max_camera_coverage, all_crimes_gdf, min_hotspot_size, strategy, graph_components=graph_components, graph_loc_keys=graph_loc_keys, timings=timings, mem_hooks={"start": start_mem, "stop": stop_mem}
    )
    
    total = timings['descriptive'] + timings['greedy'] + timings['optimal']

    ram_keys = ["descriptive_ram_peak_mb", "greedy_ram_peak_mb", "optimal_ram_peak_mb"]
    ram_values = [timings[k] for k in ram_keys if k in timings]
    if ram_values:
        timings["ram_peak_mb"] = max(ram_values)

    print(f"  + Finished in {total:.2f} seconds." )
    print(f"    - Descriptive={timings['descriptive']:.2f} seconds, Greedy={timings['greedy']:.2f} seconds, Optimal={timings['optimal']:.2f} seconds " )
    print(f"    - Peak RAM usage: {timings.get('ram_peak_mb','--'):.0f} MB" )
   
    return optimal_hotspots, greedy_hotspots, timings

def save_hotspots_csv(hotspots, out_path):
    """
    Save hotspots to CSV.
    hotspots: list of (count, (lat, lon)) in ranked order (best first).
    """
    import csv

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "crime_count", "latitude", "longitude"])

        for rank, (count, loc) in enumerate(hotspots, start=1):
            if not (isinstance(loc, tuple) and len(loc) == 2):
                raise ValueError(f"Expected (lat, lon) tuple, got: {type(loc)} {loc}")

            lat, lon = loc
            w.writerow([rank, int(count), float(lat), float(lon)])


def plot_hotspots_on_map(hotspots, html_path, zoom_start=12, radius_m=50):
    """
    Write ALL hotspots to a standalone interactive Folium HTML file.

    Supports:
      - DataFrame / GeoDataFrame with columns latitude/longitude (or geometry)
      - list of dicts
      - list of shapely Point
      - list of tuples in either form:
          A) (lat, lon) or (lat, lon, ...)
          B) (count, (lat, lon))   <-- your current format
    """

    # ---------- Normalize input ----------
    if isinstance(hotspots, pd.DataFrame):
        df = hotspots.copy()

    elif isinstance(hotspots, list):
        if not hotspots:
            raise ValueError("hotspots list is empty")

        first = hotspots[0]

        # list of dicts
        if isinstance(first, dict):
            df = pd.DataFrame(hotspots)

        # list of shapely Points
        elif isinstance(first, Point):
            df = pd.DataFrame({
                "latitude": [p.y for p in hotspots],
                "longitude": [p.x for p in hotspots],
            })

        # list of tuples
        elif isinstance(first, tuple):
            # Case B: (count, (lat, lon))
            if (
                len(first) == 2
                and isinstance(first[0], (int, float))
                and isinstance(first[1], tuple)
                and len(first[1]) == 2
            ):
                df = pd.DataFrame({
                    "total_count": [h[0] for h in hotspots],
                    "latitude":    [h[1][0] for h in hotspots],
                    "longitude":   [h[1][1] for h in hotspots],
                })

            # Case A: (lat, lon) or (lat, lon, ...)
            elif len(first) >= 2 and all(isinstance(v, (int, float)) for v in first[:2]):
                tmp = pd.DataFrame(hotspots)
                tmp = tmp.rename(columns={0: "latitude", 1: "longitude"})
                df = tmp

            else:
                raise TypeError(f"Unsupported tuple hotspot format. Example element: {first}")

        else:
            raise TypeError(f"Unsupported hotspot element type: {type(first)}")

    else:
        raise TypeError(f"Unsupported hotspots type: {type(hotspots)}")

    # ---------- Ensure lat/lon exist ----------
    if "geometry" in df.columns and ("latitude" not in df.columns or "longitude" not in df.columns):
        # GeoDataFrame case
        df["latitude"] = df.geometry.y
        df["longitude"] = df.geometry.x

    required = {"latitude", "longitude"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available columns: {list(df.columns)}")

    df["latitude"] = pd.to_numeric(df["latitude"])
    df["longitude"] = pd.to_numeric(df["longitude"])

    # Add rank if missing (rank 1 = highest count if count exists)
    if "rank" not in df.columns:
        if "total_count" in df.columns:
            df = df.sort_values("total_count", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1

    # ---------- Create map ----------
    mean_lat = float(df["latitude"].mean())
    mean_lon = float(df["longitude"].mean())
    m = folium.Map(location=[mean_lat, mean_lon], zoom_start=zoom_start)

    for _, row in df.iterrows():
        lat, lon = float(row["latitude"]), float(row["longitude"])
        rank = int(row["rank"])
        cnt = row["total_count"] if "total_count" in df.columns else None

        tooltip = f"Rank: {rank}"
        if cnt is not None:
            tooltip += f"<br>Total count: {int(cnt)}"

        # circle (in meters)
        folium.Circle(
            location=[lat, lon],
            radius=radius_m,
            color="red",
            weight=2,
            fill=True,
            fill_color="red",
            fill_opacity=0.6,
            tooltip=tooltip
        ).add_to(m)

        # rank label
        folium.Marker(
            location=[lat, lon],
            icon=folium.DivIcon(
                icon_size=(22, 22),
                icon_anchor=(11, 11),  # center of the div
                html=f"""
                <div style="
                    width: 22px;
                    height: 22px;
                    line-height: 22px;
                    border-radius: 100%;
                    text-align: center;
                    font-size: 9pt;
                    font-weight: bold;
                    color: white;
                ">
                    {rank}
                </div>
                """
            )
        ).add_to(m)

    # ---------- Save HTML ----------
    os.makedirs(os.path.dirname(html_path) or ".", exist_ok=True)
    m.save(html_path)

    return html_path