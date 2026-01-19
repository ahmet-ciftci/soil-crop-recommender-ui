#!/usr/bin/env python3
"""
soil-crop-recommender-ui - Global Training Dataset Builder
===========================================================
Scans the globe to find valid agricultural land, extracting environmental 
features and dominant crop labels for the Crop Recommendation System.
"""

import os
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from pathlib import Path
from typing import Optional, Tuple, Dict
import warnings

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_FILE = DATA_DIR / "crop_recommender_global.csv"

# Scanning parameters
TOTAL_CANDIDATES = 125_000  # Total random points to scan (~12% yield rate)
TARGET_VALID_POINTS = 15_000  # Approximate target for valid land points
NO_CROP_KEEP_RATIO = 0.05  # Keep 5% of "no crop" points labeled as NO_CROP
PROGRESS_INTERVAL = 1_000  # Print progress every N points

# File paths
ERA5_FILE = DATA_DIR / "era5_weather_2023.nc"
SOIL_PH_FILE = DATA_DIR / "soil_ph.tif"
SOIL_CLAY_FILE = DATA_DIR / "soil_clay.tif"
SOIL_SAND_FILE = DATA_DIR / "soil_sand.tif"
MAPSPAM_DIR = DATA_DIR / "spam2020V2r0_global_physical_area"

# Seed for reproducibility
RANDOM_SEED = 42


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_era5_annual_stats(era5_path: Path) -> xr.Dataset:
    """
    Load ERA5 data and compute annual statistics.
    - t2m: Annual Mean Temperature (converted from Kelvin to Celsius)
    - tp: Annual Total Precipitation (converted from meters to mm)
    """
    print("📡 Loading ERA5 weather data...")
    ds = xr.open_dataset(era5_path)
    
    # Detect time dimension name (ERA5 uses 'valid_time' or 'time')
    time_dim = None
    for dim_name in ['valid_time', 'time', 'date']:
        if dim_name in ds.dims:
            time_dim = dim_name
            break
    
    if time_dim is None:
        raise ValueError(f"Cannot find time dimension. Available dims: {list(ds.dims)}")
    
    print(f"   ✓ Time dimension: '{time_dim}' with {ds.sizes[time_dim]} steps")
    
    # Compute annual mean temperature (K -> C)
    if 't2m' in ds:
        temp_annual_mean = ds['t2m'].mean(dim=time_dim) - 273.15
    else:
        # Try alternate variable names
        temp_var = [v for v in ds.data_vars if 'temp' in v.lower() or 't2m' in v.lower()]
        if temp_var:
            temp_annual_mean = ds[temp_var[0]].mean(dim=time_dim) - 273.15
        else:
            raise ValueError("Cannot find temperature variable in ERA5 data")
    
    # Compute annual total precipitation (m -> mm)
    # Note: ERA5 monthly data often stores daily mean rates, not monthly totals
    # If 12 time steps (monthly), multiply by ~30.4 days/month to get true monthly totals
    days_per_month = 30.4 if ds.sizes[time_dim] == 12 else 1.0
    
    if 'tp' in ds:
        precip_annual_total = ds['tp'].sum(dim=time_dim) * 1000 * days_per_month
    else:
        # Try alternate variable names
        precip_var = [v for v in ds.data_vars if 'precip' in v.lower() or 'tp' in v.lower()]
        if precip_var:
            precip_annual_total = ds[precip_var[0]].sum(dim=time_dim) * 1000 * days_per_month
        else:
            raise ValueError("Cannot find precipitation variable in ERA5 data")
    
    # Create result dataset
    result = xr.Dataset({
        'temp_mean': temp_annual_mean,
        'precip_total': precip_annual_total
    })
    
    print(f"   ✓ Temperature range: {float(temp_annual_mean.min()):.1f}°C to {float(temp_annual_mean.max()):.1f}°C")
    print(f"   ✓ Precipitation range: {float(precip_annual_total.min()):.0f}mm to {float(precip_annual_total.max()):.0f}mm")
    
    return result


def sample_raster(src: rasterio.DatasetReader, lon: float, lat: float) -> Optional[float]:
    """
    Sample a single value from a raster at given coordinates.
    Returns None if the point is outside bounds or has nodata value.
    """
    try:
        # Transform geographic coordinates to pixel coordinates
        row, col = src.index(lon, lat)
        
        # Check bounds
        if row < 0 or row >= src.height or col < 0 or col >= src.width:
            return None
        
        # Read single pixel value
        window = rasterio.windows.Window(col, row, 1, 1)
        data = src.read(1, window=window)
        value = data[0, 0]
        
        # Check for nodata
        if src.nodata is not None and value == src.nodata:
            return None
        if np.isnan(value) or value < 0:
            return None
            
        return float(value)
        
    except Exception:
        return None


def sample_era5(ds: xr.Dataset, lon: float, lat: float) -> Tuple[Optional[float], Optional[float]]:
    """
    Sample temperature and precipitation from ERA5 dataset.
    Uses nearest neighbor interpolation.
    """
    try:
        temp = float(ds['temp_mean'].sel(longitude=lon, latitude=lat, method='nearest'))
        precip = float(ds['precip_total'].sel(longitude=lon, latitude=lat, method='nearest'))
        
        # Sanity checks
        if np.isnan(temp) or np.isnan(precip):
            return None, None
        if temp < -60 or temp > 60:  # Unrealistic temperature
            return None, None
        if precip < 0:
            return None, None
            
        return temp, precip
        
    except Exception:
        return None, None


def find_dominant_crop(
    crop_datasets: Dict[str, rasterio.DatasetReader],
    lon: float,
    lat: float
) -> Tuple[str, float]:
    """
    Find the crop with maximum physical area at given coordinates.
    Returns (crop_code, max_area).
    """
    max_area = 0.0
    dominant_crop = "NO_CROP"
    
    for crop_code, src in crop_datasets.items():
        area = sample_raster(src, lon, lat)
        if area is not None and area > max_area:
            max_area = area
            dominant_crop = crop_code
    
    return dominant_crop, max_area


def extract_crop_code(filename: str) -> str:
    """
    Extract crop code from MapSPAM filename.
    e.g., 'spam2020_V2r0_global_A_WHEA_A.tif' -> 'WHEA'
    """
    parts = filename.replace('.tif', '').split('_')
    # The crop code is typically the second-to-last part
    return parts[-2]


# =============================================================================
# MAIN SCANNER
# =============================================================================

def build_global_dataset():
    """
    Main function to build the global agricultural dataset.
    """
    print("=" * 60)
    print("🌍 soil-crop-recommender-ui - Global Training Dataset Builder")
    print("=" * 60)
    print()
    
    # Set random seed for reproducibility
    np.random.seed(RANDOM_SEED)
    
    # -------------------------------------------------------------------------
    # Step 1: Load ERA5 data (compute annual stats once)
    # -------------------------------------------------------------------------
    era5_ds = load_era5_annual_stats(ERA5_FILE)
    print()
    
    # -------------------------------------------------------------------------
    # Step 2: Open all raster files (keep handles open for efficiency)
    # -------------------------------------------------------------------------
    print("📂 Opening raster datasets...")
    
    # Soil files
    soil_ph_src = rasterio.open(SOIL_PH_FILE)
    soil_clay_src = rasterio.open(SOIL_CLAY_FILE)
    soil_sand_src = rasterio.open(SOIL_SAND_FILE)
    print(f"   ✓ Soil datasets loaded (pH, Clay, Sand)")
    
    # MapSPAM crop files - keep all handles open
    crop_files = sorted(MAPSPAM_DIR.glob("spam2020_V2r0_global_A_*_A.tif"))
    crop_datasets: Dict[str, rasterio.DatasetReader] = {}
    
    for crop_file in crop_files:
        crop_code = extract_crop_code(crop_file.name)
        crop_datasets[crop_code] = rasterio.open(crop_file)
    
    print(f"   ✓ MapSPAM datasets loaded ({len(crop_datasets)} crops)")
    print(f"   Crops: {', '.join(sorted(crop_datasets.keys()))}")
    print()
    
    # -------------------------------------------------------------------------
    # Step 3: Generate random coordinates
    # -------------------------------------------------------------------------
    print(f"🎲 Generating {TOTAL_CANDIDATES:,} random coordinate pairs...")
    
    # Global bounds: Lat [-60, 70] (avoiding Antarctica), Lon [-180, 180]
    lats = np.random.uniform(-60, 70, TOTAL_CANDIDATES)
    lons = np.random.uniform(-180, 180, TOTAL_CANDIDATES)
    
    print()
    
    # -------------------------------------------------------------------------
    # Step 4: The Smart Scanner Loop
    # -------------------------------------------------------------------------
    print("🔍 Starting Smart Scanner...")
    print("-" * 60)
    
    # Results storage
    results = []
    scanned = 0
    saved = 0
    skipped_ocean = 0
    skipped_desert = 0
    no_crop_count = 0
    
    for i in range(TOTAL_CANDIDATES):
        lat, lon = lats[i], lons[i]
        scanned += 1
        
        # -----------------------------------------------------------------
        # Filter 1: Ocean Check (FAST) - Sample soil pH first
        # -----------------------------------------------------------------
        ph_raw = sample_raster(soil_ph_src, lon, lat)
        
        if ph_raw is None or ph_raw < 0:
            skipped_ocean += 1
            # Progress update
            if scanned % PROGRESS_INTERVAL == 0:
                print(f"   Scanned: {scanned:,} | Saved: {saved:,} | Ocean: {skipped_ocean:,} | Desert: {skipped_desert:,}")
            continue
        
        # Convert pH (scale factor: divide by 10)
        ph = ph_raw / 10.0
        
        # Validate pH range (typical soil pH: 3.5 - 10)
        if ph < 3.0 or ph > 11.0:
            skipped_ocean += 1
            continue
        
        # -----------------------------------------------------------------
        # Extraction: Land Point - Get all features
        # -----------------------------------------------------------------
        
        # Sample other soil properties
        clay_raw = sample_raster(soil_clay_src, lon, lat)
        sand_raw = sample_raster(soil_sand_src, lon, lat)
        
        if clay_raw is None or sand_raw is None:
            skipped_ocean += 1
            continue
        
        # Convert soil properties (scale factor: divide by 10 for percentage)
        clay = clay_raw / 10.0
        sand = sand_raw / 10.0
        
        # Validate soil percentages
        if clay < 0 or clay > 100 or sand < 0 or sand > 100:
            skipped_ocean += 1
            continue
        
        # Sample ERA5 weather data
        temp, rain = sample_era5(era5_ds, lon, lat)
        
        if temp is None or rain is None:
            skipped_ocean += 1
            continue
        
        # -----------------------------------------------------------------
        # Target Check: Find dominant crop
        # -----------------------------------------------------------------
        label, max_area = find_dominant_crop(crop_datasets, lon, lat)
        
        # -----------------------------------------------------------------
        # Filter 2: Desert Check
        # -----------------------------------------------------------------
        if max_area == 0 or label == "NO_CROP":
            skipped_desert += 1
            
            # Keep a small percentage labeled as NO_CROP
            if np.random.random() < NO_CROP_KEEP_RATIO:
                no_crop_count += 1
                results.append({
                    'Lat': round(lat, 4),
                    'Lon': round(lon, 4),
                    'Temp': round(temp, 2),
                    'Rain': round(rain, 2),
                    'pH': round(ph, 2),
                    'Clay': round(clay, 2),
                    'Sand': round(sand, 2),
                    'Label': 'NO_CROP'
                })
                saved += 1
        else:
            # Valid agricultural point
            results.append({
                'Lat': round(lat, 4),
                'Lon': round(lon, 4),
                'Temp': round(temp, 2),
                'Rain': round(rain, 2),
                'pH': round(ph, 2),
                'Clay': round(clay, 2),
                'Sand': round(sand, 2),
                'Label': label
            })
            saved += 1
        
        # -----------------------------------------------------------------
        # Progress update
        # -----------------------------------------------------------------
        if scanned % PROGRESS_INTERVAL == 0:
            print(f"   Scanned: {scanned:,} | Saved: {saved:,} | Ocean: {skipped_ocean:,} | Desert: {skipped_desert:,}")
        
        # Early stopping if we have enough points
        if saved >= TARGET_VALID_POINTS * 1.5:
            print(f"\n   ✓ Target reached! Stopping early at {scanned:,} scanned points.")
            break
    
    print("-" * 60)
    print()
    
    # -------------------------------------------------------------------------
    # Step 5: Close all raster handles
    # -------------------------------------------------------------------------
    print("📁 Closing datasets...")
    soil_ph_src.close()
    soil_clay_src.close()
    soil_sand_src.close()
    for src in crop_datasets.values():
        src.close()
    era5_ds.close()
    print()
    
    # -------------------------------------------------------------------------
    # Step 6: Save results to CSV
    # -------------------------------------------------------------------------
    print("💾 Saving results...")
    
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"   ✓ Saved {len(df):,} records to: {OUTPUT_FILE}")
    print()
    
    # -------------------------------------------------------------------------
    # Step 7: Summary Statistics
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("📊 SUMMARY STATISTICS")
    print("=" * 60)
    print(f"   Total Scanned:     {scanned:,}")
    print(f"   Valid Points:      {saved:,}")
    print(f"   Skipped (Ocean):   {skipped_ocean:,}")
    print(f"   Skipped (Desert):  {skipped_desert:,}")
    print(f"   NO_CROP Labels:    {no_crop_count:,}")
    print()
    
    # Crop distribution
    print("🌾 CROP DISTRIBUTION:")
    print("-" * 40)
    crop_counts = df['Label'].value_counts()
    for crop, count in crop_counts.head(15).items():
        pct = count / len(df) * 100
        bar = "█" * int(pct / 2)
        print(f"   {crop:10s}: {count:5,} ({pct:5.1f}%) {bar}")
    
    if len(crop_counts) > 15:
        print(f"   ... and {len(crop_counts) - 15} more crops")
    print()
    
    # Feature statistics
    print("📈 FEATURE STATISTICS:")
    print("-" * 40)
    for col in ['Temp', 'Rain', 'pH', 'Clay', 'Sand']:
        print(f"   {col:6s}: min={df[col].min():8.2f}, max={df[col].max():8.2f}, mean={df[col].mean():8.2f}")
    print()
    
    print("=" * 60)
    print("✅ Dataset build complete!")
    print("=" * 60)
    
    return df


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    try:
        df = build_global_dataset()
    except FileNotFoundError as e:
        print(f"❌ Error: Required data file not found!")
        print(f"   {e}")
        print("\n   Please ensure all data files are in the /data folder.")
    except Exception as e:
        print(f"❌ Error: {e}")
        raise
