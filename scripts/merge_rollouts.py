
import pandas as pd
from pathlib import Path
import argparse
import glob

def merge_parquet_files(source_dir, output_file, chunk_size_rows):
    source_path = Path(source_dir)
    if not source_path.exists():
        print(f"Error: Directory '{source_dir}' not found.")
        return

    import json

    # Load processed files log
    processed_log_path = source_path / "processed_files.json"
    processed_files_set = set()
    if processed_log_path.exists():
        try:
            with open(processed_log_path, "r") as f:
                processed_files_set = set(json.load(f))
        except Exception as e:
            print(f"Warning: Could not load processed files log: {e}")

    # Find all parquet files matching the pattern
    all_parquet_files = list(source_path.glob("rollout_data_*.parquet"))
    
    # Filter out already processed files
    new_parquet_files = [f for f in all_parquet_files if f.name not in processed_files_set]
    
    # Sort by filename (which includes timestamp) to ensure chronological order
    new_parquet_files.sort(key=lambda x: x.name)
    
    if not new_parquet_files:
        print(f"No new 'rollout_data_*.parquet' files found in '{source_dir}'.")
        return

    print(f"Found {len(new_parquet_files)} NEW parquet files (out of {len(all_parquet_files)} total). Merging...")

    current_chunk_dfs = []
    current_chunk_rows = 0
    total_rows_processed = 0
    
    output_path_base = Path(output_file)
    output_stem = output_path_base.stem
    output_suffix = output_path_base.suffix
    output_parent = output_path_base.parent

    # Determine next chunk index
    existing_chunks = list(output_parent.glob(f"{output_stem}_part_*{output_suffix}"))
    max_index = -1
    for chunk in existing_chunks:
        try:
            # Extract index from filename "merged_rollouts_part_005.parquet"
            parts = chunk.stem.split("_part_")
            if len(parts) > 1:
                idx = int(parts[-1])
                if idx > max_index:
                    max_index = idx
        except:
            pass
    
    chunk_index = max_index + 1
    print(f"Starting at chunk index: {chunk_index}")

    def write_chunk(dfs, index):
        if not dfs:
            return 0
        merged_df = pd.concat(dfs, ignore_index=True)
        chunk_filename = f"{output_stem}_part_{index:03d}{output_suffix}"
        chunk_path = output_parent / chunk_filename
        merged_df.to_parquet(chunk_path)
        print(f"  -> Wrote chunk {index} to '{chunk_filename}' ({len(merged_df)} rows)")
        return len(merged_df)

    files_processed_in_this_run = []

    for i, f in enumerate(new_parquet_files):
        try:
            df = pd.read_parquet(f)
            # Add a column for the source filename to track origin if needed
            df["source_file"] = f.name
            
            current_chunk_dfs.append(df)
            current_chunk_rows += len(df)
            files_processed_in_this_run.append(f.name)
            
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

    # Update processed log
    processed_files_set.update(files_processed_in_this_run)
    try:
        with open(processed_log_path, "w") as f:
            json.dump(list(processed_files_set), f)
        print(f"Updated processed files log with {len(files_processed_in_this_run)} new files.")
    except Exception as e:
        print(f"Error saving processed files log: {e}")

    print(f"Successfully processed {len(new_parquet_files)} new files.")
    print(f"Total rows added: {total_rows_processed}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge rollout parquet files.")
    parser.add_argument("--dir", type=str, default="save/felix_test_training/good_runs", help="Directory containing parquet files")
    parser.add_argument("--out", type=str, default="merged_rollouts.parquet", help="Output filename base")
    parser.add_argument("--chunk-size", type=int, default=100000, help="Number of rows per chunk file")
    
    args = parser.parse_args()
    
    merge_parquet_files(args.dir, args.out, args.chunk_size)
