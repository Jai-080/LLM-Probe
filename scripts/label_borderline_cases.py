import os
import sys
import re
import pandas as pd
import numpy as np

def longest_common_ngram_len(text1: str, text2: str) -> int:
    """
    Computes the word-level longest common contiguous sequence of words (n-gram).
    """
    if pd.isna(text1) or pd.isna(text2):
        return 0
    words1 = str(text1).split()
    words2 = str(text2).split()
    if not words1 or not words2:
        return 0
    m, n = len(words1), len(words2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    longest = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if words1[i - 1].lower() == words2[j - 1].lower():
                dp[i][j] = dp[i - 1][j - 1] + 1
                longest = max(longest, dp[i][j])
            else:
                dp[i][j] = 0
    return longest

def get_longest_common_ngram(text1: str, text2: str) -> str:
    """
    Finds the actual longest common contiguous sequence of words.
    Matches the logic of longest_common_ngram_len but returns the matched substring.
    """
    if pd.isna(text1) or pd.isna(text2):
        return ""
    words1 = str(text1).split()
    words2 = str(text2).split()
    if not words1 or not words2:
        return ""
    m, n = len(words1), len(words2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    longest = 0
    end_i = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if words1[i - 1].lower() == words2[j - 1].lower():
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > longest:
                    longest = dp[i][j]
                    end_i = i
            else:
                dp[i][j] = 0
    if longest == 0:
        return ""
    return " ".join(words1[end_i - longest : end_i])

def has_distinctive_constant(matched_str: str) -> tuple:
    """
    Checks if the matched string contains a numeric literal of 3+ digits,
    or a hex/hash-like string of 8+ chars. Identifier checking is dropped
    to avoid false positives on common/idiomatic variable and method names.
    Returns (True, reason) or (False, None).
    """
    if not matched_str:
        return False, None
        
    # 1. Numeric literal of 3+ digits
    num_match = re.search(r'\b\d{3,}\b', matched_str)
    if num_match:
        return True, f"Numeric literal: '{num_match.group()}'"
        
    # 2. Hex/hash-like string (8+ characters)
    hex_match = re.search(r'\b[0-9a-fA-F]{8,}\b', matched_str)
    if hex_match:
        return True, f"Hex/Hash: '{hex_match.group()}'"
        
    return False, None

def main():
    csv_path = "data/memorization/labeled/needs_manual_review.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        sys.exit(1)
        
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    
    labels = []
    reasons = []
    
    # Statistical counters
    stats = {}
    
    for idx, row in df.iterrows():
        cat = row['source_category']
        token_count = row['matched_token_count']
        ngram_len = row['longest_common_ngram_len']
        gen_text = row['generated_continuation']
        true_text = row['true_continuation']
        
        if cat not in stats:
            stats[cat] = {
                'total': 0, 'memorized': 0, 'novel': 0,
                'token_rule': 0, 'ngram_rule': 0, 'constant_rule': 0
            }
        stats[cat]['total'] += 1
        
        assigned_label = 'novel'
        reason = "Default novel classification"
        
        if cat in ['book_openings', 'lyrics']:
            if token_count >= 10:
                assigned_label = 'memorized'
                reason = f"matched_token_count ({token_count}) >= 10"
                stats[cat]['token_rule'] += 1
            elif ngram_len >= 8:
                assigned_label = 'memorized'
                reason = f"longest_common_ngram_len ({ngram_len}) >= 8"
                stats[cat]['ngram_rule'] += 1
                
        elif cat == 'wiki_leads':
            if token_count >= 10 and ngram_len >= 6:
                assigned_label = 'memorized'
                reason = f"matched_token_count ({token_count}) >= 10 AND ngram_len ({ngram_len}) >= 6"
                stats[cat]['token_rule'] += 1
                
        elif cat == 'code_snippets':
            if ngram_len >= 30:
                assigned_label = 'memorized'
                reason = f"longest_common_ngram_len ({ngram_len}) >= 30"
                stats[cat]['ngram_rule'] += 1
            else:
                matched_span = get_longest_common_ngram(gen_text, true_text)
                has_const, const_reason = has_distinctive_constant(matched_span)
                if has_const:
                    assigned_label = 'memorized'
                    reason = f"Distinctive constant found: {const_reason}"
                    stats[cat]['constant_rule'] += 1
                    
        elif cat == 'novel_prompts':
            assigned_label = 'novel'
            reason = "Novel prompt (invented scenarios are always novel)"
            
        if assigned_label == 'memorized':
            stats[cat]['memorized'] += 1
        else:
            stats[cat]['novel'] += 1
            
        labels.append(assigned_label)
        reasons.append(reason)
        
    df['label'] = labels
    df['assigned_reason'] = reasons
    
    # Save the labeled output
    output_path = "data/memorization/labeled/needs_manual_review_labeled.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved fully re-labeled borderline set to: {output_path}")
    
    # Print rule-based summary
    print("\n" + "="*60)
    print(" RULE-BASED LABELING SUMMARY")
    print("="*60)
    for cat, data in stats.items():
        print(f"Category: {cat} (Total: {data['total']})")
        print(f"  - MEMORIZED: {data['memorized']} | NOVEL: {data['novel']}")
        print(f"  - Token Match Rule: {data['token_rule']}")
        print(f"  - N-gram Overlap Rule: {data['ngram_rule']}")
        print(f"  - Constant Detection Rule: {data['constant_rule']}")
        print("-" * 40)
    print("="*60 + "\n")
    
    # Boundary Selection for Spotcheck
    boundary_rows = []
    
    for idx, row in df.iterrows():
        cat = row['source_category']
        token_count = row['matched_token_count']
        ngram_len = row['longest_common_ngram_len']
        
        is_boundary = False
        if cat in ['book_openings', 'lyrics']:
            if 8 <= token_count <= 12 or 6 <= ngram_len <= 10:
                is_boundary = True
        elif cat == 'wiki_leads':
            if 8 <= token_count <= 12 and 4 <= ngram_len <= 8:
                is_boundary = True
        elif cat == 'code_snippets':
            if 10 <= ngram_len <= 14:
                is_boundary = True
                
        if is_boundary:
            boundary_rows.append(row)
            
    boundary_df = pd.DataFrame(boundary_rows)
    
    # Select sample
    target_sample_size = min(30, len(df))
    if len(boundary_df) >= target_sample_size:
        sample_df = boundary_df.sample(n=target_sample_size, random_state=42)
    else:
        # Top-up with other rows if not enough boundary rows exist
        rem_count = target_sample_size - len(boundary_df)
        remaining_indices = [i for i in df.index if i not in boundary_df.index]
        rem_df = df.loc[np.random.choice(remaining_indices, size=min(rem_count, len(remaining_indices)), replace=False)]
        sample_df = pd.concat([boundary_df, rem_df], ignore_index=True)
        
    spotcheck_path = "data/memorization/labeled/borderline_spotcheck_sample.csv"
    sample_df.to_csv(spotcheck_path, index=False)
    print(f"Saved {len(sample_df)} spotcheck rows to: {spotcheck_path}")

if __name__ == "__main__":
    main()
