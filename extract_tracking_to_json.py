#!/usr/bin/env python3
"""
Extract tracking data from TrackLab .pklz file and examine DataFrame structure.
"""

import sys
import zipfile
import pickle
import json
from pathlib import Path


def load_and_examine_tracker_state(pklz_path, output_json_path):
    """Load tracker state and examine its structure."""
    
    print(f"Loading tracker state from: {pklz_path}")
    
    # Load the .pklz file (it's a ZIP archive)
    with zipfile.ZipFile(pklz_path, 'r') as zf:
        print(f"Files in archive: {zf.namelist()}")
        
        # Load the main pickle file (not the JSON summary)
        pkl_files = [f for f in zf.namelist() if f.endswith('.pkl') and not 'image' in f]
        
        if not pkl_files:
            print("❌ No .pkl files found in archive")
            return
        
        pkl_file = pkl_files[0]
        print(f"Loading: {pkl_file}")
        
        with zf.open(pkl_file) as f:
            tracker_state = pickle.load(f)
    
    print(f"✅ Tracker state loaded successfully")
    print(f"Type: {type(tracker_state)}")
    
    # It's a DataFrame - examine its structure
    import pandas as pd
    if isinstance(tracker_state, pd.DataFrame):
        print(f"\n📊 DataFrame Info:")
        print(f"  Shape: {tracker_state.shape} (rows x columns)")
        print(f"  Columns: {list(tracker_state.columns)}")
        print(f"  Index name: {tracker_state.index.name if tracker_state.index.name else 'unnamed'}")
        
        # Check if it's a multi-index DataFrame (common in TrackLab)
        if isinstance(tracker_state.index, pd.MultiIndex):
            print(f"\n📑 MultiIndex DataFrame:")
            print(f"  Index levels: {tracker_state.index.names}")
            for i, level_name in enumerate(tracker_state.index.names):
                unique_vals = tracker_state.index.get_level_values(i).unique()
                print(f"  Level {i} ({level_name}): {len(unique_vals)} unique values")
                print(f"    First 10: {unique_vals[:10].tolist()}")
        else:
            print(f"\n📑 Single Index:")
            print(f"  Unique values: {tracker_state.index.nunique()}")
            print(f"  First 10: {tracker_state.index[:10].tolist()}")
        
        print(f"\n📋 First 10 rows:")
        print(tracker_state.head(10))
        
        print(f"\n📋 Data types:")
        print(tracker_state.dtypes)
        
        # Try to extract tracking data to JSON
        print(f"\n🔄 Attempting to convert to JSON format...")
        
        try:
            # Reset index to make it regular columns
            df_reset = tracker_state.reset_index()
            
            # Convert to records format
            records = df_reset.to_dict('records')
            
            print(f"✅ Converted {len(records)} records")
            
            # Save to JSON
            output_data = {
                'metadata': {
                    'source': str(pklz_path),
                    'num_records': len(records),
                    'columns': list(df_reset.columns)
                },
                'tracks': records
            }
            
            with open(output_json_path, 'w') as f:
                json.dump(output_data, f, indent=2, default=str)
            
            print(f"✅ Saved tracking data to: {output_json_path}")
            
            # Show sample record
            if records:
                print(f"\n📝 Sample record:")
                print(json.dumps(records[0], indent=2, default=str))
                
        except Exception as e:
            print(f"❌ Error converting to JSON: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"⚠️ Not a DataFrame, examining attributes...")
        print(f"Available attributes: {[a for a in dir(tracker_state) if not a.startswith('_')][:30]}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_tracking_to_json.py <input.pklz> <output.json>")
        sys.exit(1)
    
    pklz_path = sys.argv[1]
    output_json = sys.argv[2]
    
    load_and_examine_tracker_state(pklz_path, output_json)
