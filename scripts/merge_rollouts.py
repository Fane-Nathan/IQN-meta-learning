
import pandas as pd
from pathlib import Path
import argparse
import glob

def merge_parquet_files(source_dir, output_file, chunk_size_rows):
    source_path = Path(source_dir)
    if not source_path.exists():
        print(f"Error: Directory '{source_dir}' not found.")
        return

    # Find all parquet files matching the pattern
    parquet_files = list(source_path.glob("rollout_data_*.parquet"))
    # Sort by filename (which includes timestamp) to ensure chronological order
    parquet_files.sort(key=lambda x: x.name)
    
    if not parquet_files:
        print(f"No 'rollout_data_*.parquet' files found in '{source_dir}'.")
        return

    print(f"Found {len(parquet_files)} parquet files. Merging with chunk size {chunk_size_rows} rows...")

    current_chunk_dfs = []
    current_chunk_rows = 0
    chunk_index = 0
    total_rows_processed = 0
    
    output_path_base = Path(output_file)
    output_stem = output_path_base.stem
    output_suffix = output_path_base.suffix
    output_parent = output_path_base.parent

    def write_chunk(dfs, index):
        if not dfs:
            return 0
        merged_df = pd.concat(dfs, ignore_index=True)
        chunk_filename = f"{output_stem}_part_{index:03d}{output_suffix}"
        chunk_path = output_parent / chunk_filename
        merged_df.to_parquet(chunk_path)
        print(f"  -> Wrote chunk {index} to '{chunk_filename}' ({len(merged_df)} rows)")
        return len(merged_df)

    for i, f in enumerate(parquet_files):
        try:
            df = pd.read_parquet(f)
            # Add a column for the source filename to track origin if needed
            df["source_file"] = f.name
            
            current_chunk_dfs.append(df)
            current_chunk_rows += len(df)
            
            if current_chunk_rows >= chunk_size_rows:
                total_rows_processed += write_chunk(current_chunk_dfs, chunk_index)
                chunk_index += 1
                current_chunk_dfs = []
                current_chunk_rows = 0
                
        except Exception as e:
            print(f"Warning: Could not read {f}: {e}")

    # Write remaining data
    if current_chunk_dfs:
        total_rows_processed += write_chunk(current_chunk_dfs, chunk_index)

    print(f"Successfully processed {len(parquet_files)} files.")
    print(f"Total rows: {total_rows_processed}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge rollout parquet files.")
    parser.add_argument("--dir", type=str, default="save/felix_test_training/good_runs", help="Directory containing parquet files")
    parser.add_argument("--out", type=str, default="merged_rollouts.parquet", help="Output filename base")
    parser.add_argument("--chunk-size", type=int, default=100000, help="Number of rows per chunk file")
    
    args = parser.parse_args()
    
    merge_parquet_files(args.dir, args.out, args.chunk_size)
