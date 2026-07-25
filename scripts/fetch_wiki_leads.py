import os
import re
import sys
import time
import urllib.parse
import logging
import requests
from typing import Set

# Set up logging to stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
TOPICS_FILE = "data/memorization/raw/wiki_topics.txt"
WIKI_LEADS_DIR = "data/memorization/raw/wiki_leads"
USER_AGENT = "LLMProbeResearchBot/1.0 (academic research; contact: project-llm-probe@example.com)"
RATE_LIMIT_DELAY = 0.5  # seconds
MIN_WORD_COUNT = 30


def sanitize_filename(title: str) -> str:
    """
    Sanitizes a page title to be a safe Windows filename.
    Replaces spaces with underscores and invalid chars with underscores.
    """
    # Keep only alphanumeric, spaces, hyphens, and underscores, replacing others with underscore
    clean = re.sub(r'[\\/*?:"<>|]', '_', title)
    clean = clean.replace(' ', '_')
    # Deduplicate underscores
    clean = re.sub(r'_+', '_', clean)
    return clean.strip('_')


def clean_summary_text(text: str) -> str:
    """
    Collapses whitespace and strips citation brackets (e.g., [1], [citation needed]).
    """
    # Collapse multiple whitespace characters/newlines to a single space
    cleaned = re.sub(r'\s+', ' ', text).strip()
    # Strip numeric citation markers like [1], [12], etc.
    cleaned = re.sub(r'\[\d+\]', '', cleaned)
    # Strip common text citations like [citation needed]
    cleaned = re.sub(r'\[citation needed\]', '', cleaned)
    # Re-collapse whitespace in case removing citations left extra spaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def main():
    logger.info("=== Starting Wikipedia Lead Paragraph Fetching Utility ===")
    
    # 1. Setup Directories
    os.makedirs(WIKI_LEADS_DIR, exist_ok=True)
    
    if not os.path.exists(TOPICS_FILE):
        logger.error(f"Topics file not found at: {TOPICS_FILE}")
        sys.exit(1)
        
    # 2. Read Topics
    with open(TOPICS_FILE, "r", encoding="utf-8") as f:
        topics = [line.strip() for line in f if line.strip()]
        
    logger.info(f"Loaded {len(topics)} topics from {TOPICS_FILE}")
    
    # Counter metrics
    total_fetched = 0
    total_skipped_too_short = 0
    total_skipped_disambiguation = 0
    total_failed = 0
    total_already_existing = 0
    
    headers = {
        "User-Agent": USER_AGENT
    }
    
    # 3. Process each topic
    for idx, topic in enumerate(topics, 1):
        sanitized = sanitize_filename(topic)
        filename = f"{sanitized}.txt"
        filepath = os.path.join(WIKI_LEADS_DIR, filename)
        
        # Resumability check
        if os.path.exists(filepath):
            logger.info(f"[{idx}/{len(topics)}] [EXISTING] Skipping '{topic}' (File already exists: {filename})")
            total_already_existing += 1
            continue
            
        # URL encode topic title
        encoded_title = urllib.parse.quote(topic.replace(' ', '_'))
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded_title}"
        
        try:
            logger.info(f"[{idx}/{len(topics)}] [FETCHING] Getting summary for '{topic}'...")
            response = requests.get(url, headers=headers, timeout=10)
            
            # Rate limiting
            time.sleep(RATE_LIMIT_DELAY)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check for disambiguation page
                page_type = data.get("type", "standard")
                if page_type == "disambiguation":
                    logger.warning(f"[{idx}/{len(topics)}] [SKIP] '{topic}' resolved to a disambiguation page. Skipping.")
                    total_skipped_disambiguation += 1
                    continue
                    
                extract = data.get("extract", "")
                cleaned_text = clean_summary_text(extract)
                
                # Check word count
                word_count = len(cleaned_text.split())
                if word_count < MIN_WORD_COUNT:
                    logger.warning(
                        f"[{idx}/{len(topics)}] [SKIP] Summary for '{topic}' is too short "
                        f"({word_count} words, min {MIN_WORD_COUNT} required). Skipping."
                    )
                    total_skipped_too_short += 1
                    continue
                    
                # Save to file
                with open(filepath, "w", encoding="utf-8") as out_f:
                    out_f.write(cleaned_text + "\n")
                    
                logger.info(f"[{idx}/{len(topics)}] [SUCCESS] Saved '{topic}' summary ({word_count} words) to {filename}")
                total_fetched += 1
                
            elif response.status_code == 404:
                logger.error(f"[{idx}/{len(topics)}] [FAIL] 404 Not Found for topic '{topic}'. Check if spelling matches Wikipedia.")
                total_failed += 1
            else:
                logger.error(f"[{idx}/{len(topics)}] [FAIL] HTTP {response.status_code} for topic '{topic}'.")
                total_failed += 1
                
        except requests.RequestException as e:
            logger.error(f"[{idx}/{len(topics)}] [ERROR] Request failed for topic '{topic}': {e}")
            total_failed += 1
            # Add sleep in case of connection dropouts to avoid aggressive looping
            time.sleep(RATE_LIMIT_DELAY)
            
    # Final Reporting
    print("\n" + "=" * 40)
    print(" WIKIPEDIA LEAD FETCH COMPLETE")
    print("=" * 40)
    print(f"Total Topics Configured: {len(topics)}")
    print(f"  - Already Existing (skipped): {total_already_existing}")
    print(f"  - Successfully Fetched:       {total_fetched}")
    print(f"  - Skipped (disambiguation):   {total_skipped_disambiguation}")
    print(f"  - Skipped (too short):        {total_skipped_too_short}")
    print(f"  - Failed (errors/404):        {total_failed}")
    print("=" * 40 + "\n")


if __name__ == '__main__':
    main()
