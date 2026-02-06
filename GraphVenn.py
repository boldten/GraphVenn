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
import pandas as pd
import geopandas as gpd
import os
import time
import psutil
import pickle
from datetime import datetime, timezone
import gc
import argparse

import GraphVenn_functions as gv # GraphVenn's helper functions


data_path = '../../Data/'
results_path = 'Results/'

if __name__ == "__main__":
    gv.show_disclaimer()

    parser = argparse.ArgumentParser(description="Run GraphVenn with specified parameters.")

    # Required named arguments
    required_named = parser.add_argument_group('required named arguments')
    required_named.add_argument("--csv", required=True, help="Path to the input crime data CSV file")
    required_named.add_argument("--city", required=True, type=str, help="City for slicing data")

    # Optional arguments with defaults
    parser.add_argument("--strategy", default='greedy', type=str, help="The strategy used for hotspot dection. Either 'greedy' or 'optimal', where 'greedy' gives fast near optimal result. While 'optimal' uses ILP to find optimal solution, but also orders of magnitude slower than 'greedy'.")
    parser.add_argument("--N", type=int, default=100, help="Number of hotspots to detect. When using strategy='optimal' then generally keep N at 100 or lower. For 'greedy' it can be higher.")
    parser.add_argument("--d", type=int, default=100, help="Max allowed crime distance (in meters) from hotspot centers, default = 100")
    parser.add_argument("--p", type=int, default=4, help="Spatial resolution as number of decimals in Lon/Lat coordinates to use, generally keep at 4 or lower.")
    parser.add_argument("--min_hotspot_count", type=int, default=1, help="Minimum number of crimes that must be included in a potential hotspot.")
    parser.add_argument("--verbose", type=int, choices=[0, 1, 2], default=1, help="Verbosity level: 0=Basic, 1=Progress, 2=Debug")

    args = parser.parse_args()

    # Assign arguments to variables
    crime_data_path = args.csv
    city = args.city
    strategy = args.strategy
    N = args.N
    d = args.d
    p = args.p
    min_hotspot_count = args.min_hotspot_count
    verbose = args.verbose
    gv.verbose = args.verbose

    # Output frame 
    time_cols = [
        'City','Radius_m','Precision_decimals','N','Crimes','Unique_Positions','Crimes_per_Unique',
        'Execution_time_Descriptive_s','Execution_time_Greedy_s','Execution_time_Optimal_s',
        'RAM_Peak_MB','Timestamp','RAM_descriptive_MB', 'RAM_greedy_MB', 'RAM_optimal_MB'
    ]
    time_performance_df = pd.DataFrame(columns=time_cols)

    # Check if directory exists, otherwise create them
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Memory tracking
    try:
        _ps = psutil.Process()
        have_psutil = True
    except Exception:
        have_psutil = False
    def current_mem_mb():
        if have_psutil:
            return _ps.memory_info().rss / (1024*1024)
        return float('nan')
    gv.check_csv_file( crime_data_path )

    print(f"==============================================[ City: {city} ]==============================================")
    unique_position_crime_counts, all_crimes_gdf = gv.create_geodataframe( crime_data_path, False )
    pai_A = gv.calculate_area_gdf( all_crimes_gdf ) # Calculate the full study area using a rectangle that includes all crimes
    pai_a = gv.calculate_camera_area(d)               # circle area (km^2)
    gv.print_verbose( 0, "x Study area (km^2) = {:>6}".format( round(pai_A,0) ) )    

    # ---- Save dataframes to file for later use (robust to different df shapes) ----
    out_dir = "Results/Processed"
    os.makedirs(out_dir, exist_ok=True)

    # Build unique positions GeoDataFrame
    unique_gdf = gv.to_unique_gdf(unique_position_crime_counts)

    # Ensure all_crimes_gdf has CRS and is GeoDataFrame
    if not isinstance(all_crimes_gdf, gpd.GeoDataFrame):
        all_crimes_gdf = gpd.GeoDataFrame(all_crimes_gdf, geometry="geometry", crs="EPSG:4326")
    elif all_crimes_gdf.crs is None:
        all_crimes_gdf.set_crs(epsg=4326, inplace=True, allow_override=True)

    # Filepaths use the loop's city
    unique_fp = os.path.join(out_dir, f"{city}_unique_positions.gpkg")
    crimes_fp = os.path.join(out_dir, f"{city}_all_crimes.gpkg")

    unique_gdf.to_file(unique_fp, layer="unique", driver="GPKG")
    all_crimes_gdf.to_file(crimes_fp, layer="all_crimes", driver="GPKG")
    # ---- end save block ----

    # Descriptives
    num_crimes = int(all_crimes_gdf.shape[0])
    gv.print_verbose( 0, "x Number of crimes  = {:>6}".format( num_crimes ) )
    num_unique = int(unique_position_crime_counts.shape[0])
    gv.print_verbose( 0, "x Unique positions  = {:>6}".format( num_unique ) )
    ratio = (num_crimes / num_unique) if num_unique > 0 else float('nan')
    gv.print_verbose( 0, "x Crimes / Uniq.pos = {:>6.3f}\n".format( ratio ) )   

    # Map p to your method tag (kept for consistent printing/logging)
    run_times = []
    mem_samples = []
    t0 = time.time()
    mem0 = current_mem_mb()

    # Core call (this is what Part I times):
    optimal_hotspots, greedy_hotspots, measurements = gv.run_graphvenn_method(
        unique_position_crime_counts.copy(),
        d,
        p,
        N,
        all_crimes_gdf,
        min_hotspot_count,
        strategy=strategy
    )

    if strategy in {"both", "optimal"}:
        counts = [cnt for cnt, _ in optimal_hotspots]
    if strategy in {"both", "greedy"}:
        counts = [cnt for cnt, _ in greedy_hotspots]

    # -------- save hotspot results to pickle file for this iteration --------
    if strategy in {"both", "greedy"}:
        result_csv = os.path.join(
            results_path,
            f"GraphVenn_result_top{N}_greedy_{city}_d={d}_p={p}.csv"
        )
        gv.save_hotspots_csv(greedy_hotspots, result_csv)

    if strategy in {"both", "optimal"}:
        result_csv = os.path.join(
            results_path,
            f"GraphVenn_result_top{N}_optimal_{city}_d={d}_p={p}.csv"
        )
        gv.save_hotspots_csv(optimal_hotspots, result_csv)

    gc.collect()