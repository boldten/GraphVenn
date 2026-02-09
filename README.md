# *GraphVenn*: A Globally Optimal Algorithm for Hotspot Detection and Ranking

**GraphVenn** is a Python implementation of a globally optimal crime hotspot detection and ranking method based on spatial optimization. The algorithm formulates hotspot selection as a constrained optimization problem and solves it using integer linear programming (ILP), ensuring optimal solutions under the specified model assumptions. However, it also supports a fast greedy approximate hotspot detection strategy.

GraphVenn operates on *unique spatial locations* (aggregated latitude–longitude coordinates) rather than individual crime events. This representation substantially reduces problem size while preserving exact crime counts per location, enabling scalable analysis on large urban datasets.

To improve efficiency, GraphVenn combines:
- spatial preprocessing and pruning,
- graph-based decomposition of independent subproblems, and
- exact optimization using the CBC MILP solver via PuLP.

The result is a method that provides **provably optimal hotspot selections**, suitable for research and comparative evaluation of hotspot detection strategies.

For full details, see the article *A Globally Optimal Algorithm for Hotspot Detection and Ranking* in [*Crime Science*](????).


![GraphVenn runtime demo](docs/graphvenn_demo.gif)

---

## Project Structure

```
.
├── GraphVenn.py               	# Main entry point to run the hotspot detection pipeline
├── GraphVenn_functions.py     	# Core logic: graph construction, hotspot ranking, evaluation
├── GraphVenn_ilp.py     		# Integer linear programming (ILP) routines for optimal hotspot selection
├── GraphVenn_mem.py     		# Lightweight memory profiling utilities used during algorithm phases
├── GraphVenn_bloom.py     		# Compact Bloom-filter–based dominance prechecks for pruning candidates
├── LICENSE 	               	# MIT License (see below)
├── Data/                      	# Input data directory (crime datasets as CSV)
│   └── NYC_2016_crimes.csv 	# Example crime data from New York City, 489k crimes in 2016. From the CODE open crime database.
└── Results/                   	# Output directory (CSV and HTML result files)
```

---

## Requirements

### Python dependencies
- numpy
- pandas
- geopandas
- shapely
- psutil
- pulp

Install via:

```bash
pip install -r requirements.txt
```

### System dependencies
GraphVenn uses the *CBC* MILP solver via *PuLP*. CBC must be installed and available on your system PATH.

Install using *Conda / Miniforge* (recommended):
```bash
conda install -c conda-forge coincbc
```

Or in *macOS* using Homebrew:
```bash
brew install cbc
```

---

## Usage

Run GraphVenn from the command line with the required arguments `--csv` and `--city`, and optionally override default parameters:

```bash
python GraphVenn.py --csv <path_to_csv_file> --city <city_name> [other options...]
```

### Required Arguments

- `--csv` — Path to the input crime data CSV file.
- `--city` — Name of the city to analyze (used for labeling and filtering).

### Optional Parameters

| Argument             | Default   | Description 
|----------------------|-----------|-------------
| `--strategy`		   | `'greedy'`| Search strategy, either `greedy` or `optimal`.
| `--N` 			   | `100`     | Only consider top N positions with most crimes. Use `-1` to include all.
| `--d`    			   | `100`     | Hotspot radius in meters.
| `--p`			       | `4`       | Spatial resolution in terms of number of coordinate decimals.
| `--min_cluster_size` | `1`       | Minimum number of crimes required at a location to be considered.
| `--verbose`          | `1`       | Verbosity level: `0` = minimal, `1` = progress info, `2` = debug details.

Run with default values and synthetic example data in directory ./Data:

```bash
python GraphVenn.py --csv Data/NYC_2016_crimes.csv --city NYC
```

Run on the same data but for 100 hotspots (N), 50m radius (d), a spatial resolution of 4 decimals (p), and again the 'greedy' detection strategy.

```bash
python GraphVenn.py --csv Data/NYC_2016_crimes.csv --city NYC --N 100 --d 50 --p 4 --strategy='greedy'
```

Run on the same data but for 50 hotspots (N), 100m radius (d), a spatial resolution of 4 decimals (p), but now with the 'optimal' detection strategy relying on ILP.

```bash
python GraphVenn.py --csv Data/NYC_2016_crimes.csv --city NYC --N 50 --d 100 --p 4 --strategy='optimal'
```

---

## Data input

The program takes one argument that should be the path to a CSV file, comma (,) separated, containing the crime data to analyze. The columns in the CSV should be named:
- crime_code: integer values representing the type of crime. In the example crime data file it is set to a dummy value (99) for all crimes.
- year: integer values representing the year when the crime occurred, e.g., 2025
- latitude:  floating point values representing latitude coordinates, e.g., 55.6050 (note the dot, not a comma)
- longitude: floating point values representing longitude coordinates, e.g., 13.1075

---

## Output format

GraphVenn stores detected hotspot results as a **CSV file** and an **interactive HTML map** in the `Results/` directory.

Each CSV row corresponds to **one hotspot candidate**, ordered by descending priority:
- **Row 1** contains the *best-ranked* hotspot.
- Subsequent rows list hotspots in decreasing order of importance.

Each row includes the hotspot's **rank**, its **crime count** (with crimes assigned so they are not double-counted across hotspots), and the hotspot center **latitude/longitude** coordinates.

The interactive map allows zooming and panning to inspect hotspot locations. It is generated using **OpenStreetMap** tiles via the **Folium** package.

### CSV columns

| Column name   | Description |
|---------------|-------------|
| `rank`        | Rank of the hotspot (1 = best). |
| `crime_count` | Number of crimes associated with the hotspot location. |
| `latitude`    | Latitude of the hotspot center (decimal degrees). |
| `longitude`   | Longitude of the hotspot center (decimal degrees). |

### Example result output

```csv
rank,crime_count,latitude,longitude
1,1765,40.6798,-73.7763
2,1735,40.7908,-73.8847
3,1451,40.75,-73.9894
```

---

## Disclaimer

This code is intended for research and educational purposes only. It is not intended for operational use or for real-time law enforcement decisions without independent validation and oversight. Use at your own risk.

---

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

## Citation

If you use GraphVenn in your work, please cite the following publication:

> Boldt, M. (2026). *A Globally Optimal Algorithm for Hotspot Detection and Ranking*. *Crime Science*, 1-?. [???https://doi.org/10.1007/s10940-025-09623-9](???https://doi.org/10.1007/s10940-025-09623-9)

BibTeX:
```bibtex
@article{boldt2026graphvenn,
author = {Boldt, Martin},
doi = {???},
journal = {Crime Science},
month = dec,
title = {{A Globally Optimal Algorithm for Hotspot Detection and Ranking}},
pages = {1-??},
url = {???},
year = {2026}
}
```

---