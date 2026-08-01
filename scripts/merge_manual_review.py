import os
import sys
import logging
import pandas as pd
import numpy as np

# Include parent workspace folder in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.build_memorization_dataset import is_duplicate_tfidf

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parquet_path = "data/memorization/labeled/dataset.parquet"
    csv_path = "data/memorization/labeled/needs_manual_review_labeled.csv"
    output_path = "data/memorization/labeled/dataset_v2.parquet"
    
    if not os.path.exists(parquet_path):
        logger.error(f"Base dataset not found at: {parquet_path}")
        sys.exit(1)
    if not os.path.exists(csv_path):
        logger.error(f"Re-labeled manual review file not found at: {csv_path}")
        sys.exit(1)
        
    # 1. Load files
    df_pq = pd.read_parquet(parquet_path)
    df_csv = pd.read_csv(csv_path)
    
    logger.info(f"Loaded base dataset: {len(df_pq)} rows")
    logger.info(f"Loaded newly labeled CSV: {len(df_csv)} rows")
    
    # Print baseline counts
    print("\n=== Baseline dataset.parquet counts ===")
    print(df_pq['label'].value_counts())
    print(df_pq['source_category'].value_counts())
    
    # 2. Align schema
    # Keep only columns that exist in dataset.parquet
    target_columns = df_pq.columns.tolist()
    
    # Print CSV labels count
    print("\n=== CSV needs_manual_review_labeled.csv counts ===")
    print(df_csv['label'].value_counts())
    
    # 3. Check for source_id overlap & duplicates
    existing_ids = set(df_pq['source_id'].dropna().tolist())
    
    merged_rows = []
    id_duplicates_count = 0
    tfidf_duplicates_count = 0
    
    # Initialize TF-IDF comparison list from base dataset
    existing_combined_texts = (df_pq['prompt'] + " " + df_pq['generated_continuation']).tolist()
    
    logger.info("Iterating through CSV rows and running deduplication checks...")
    for idx, row in df_csv.iterrows():
        source_id = row['source_id']
        
        # Check source_id overlap
        if source_id in existing_ids:
            logger.warning(f"Excluding row {source_id} due to exact source_id overlap with main dataset.")
            id_duplicates_count += 1
            continue
            
        # Check TF-IDF cosine similarity
        combined_candidate = str(row['prompt']) + " " + str(row['generated_continuation'])
        if is_duplicate_tfidf(combined_candidate, existing_combined_texts):
            logger.warning(f"Excluding row {source_id} due to TF-IDF similarity (>0.95) with existing sequences.")
            tfidf_duplicates_count += 1
            continue
            
        # Add to comparison pool to avoid duplicates inside the CSV itself
        existing_combined_texts.append(combined_candidate)
        
        # Select and align columns
        aligned_row = row[target_columns].to_dict()
        merged_rows.append(aligned_row)
        
    # 4. Concatenate and Merge
    if merged_rows:
        df_new_rows = pd.DataFrame(merged_rows)
        # Ensure column types are aligned
        for col in target_columns:
            df_new_rows[col] = df_new_rows[col].astype(df_pq[col].dtype)
        df_v2 = pd.concat([df_pq, df_new_rows], ignore_index=True)
    else:
        df_v2 = df_pq.copy()
        logger.info("No new non-duplicate rows found to merge.")
        
    # 5. Save Output
    df_v2.to_parquet(output_path, index=False)
    logger.info(f"Successfully created merged dataset version 2 at: {output_path}")
    
    # Print comparison metrics
    print("\n" + "="*50)
    print(" DATASET MERGE & DEDUPLICATION SUMMARY")
    print("="*50)
    print(f"Original dataset.parquet: {len(df_pq)} rows")
    print(f"Newly added from CSV    : {len(merged_rows)} rows")
    print(f"Source ID duplicates    : {id_duplicates_count} dropped")
    print(f"TF-IDF similarity dups  : {tfidf_duplicates_count} dropped")
    print(f"Final dataset_v2.parquet: {len(df_v2)} rows")
    print("-" * 50)
    print("Final Label Distribution:")
    print(df_v2['label'].value_counts())
    print("\nFinal Category Distribution:")
    print(df_v2['source_category'].value_counts())
    print("="*50)
    
    # 6. Explicit warning notification
    print("\n[IMPORTANT NOTICE]")
    print("Alice's Adventures in Wonderland and other previously-held-out examples")
    print("are now part of the mergeable pool in dataset_v2.parquet.")
    print("If you retrain a probe using this new parquet, you must check")
    print("and log which split (TRAIN or TEST) these files land in before")
    print("citing them as out-of-sample generalization evidence in reports.")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
