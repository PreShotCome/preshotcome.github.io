import argparse
import json
import logging
import os
import re
import requests
import time
import uuid
from urllib.parse import urlparse

# Configure logging with a default level
def setup_logging(log_level):
    """Setup logging with specified level"""
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('chan_downloader.log'),
            logging.StreamHandler()
        ]
    )

# Statistics tracking
class DownloadStats:
    def __init__(self):
        self.threads_processed = 0
        self.threads_skipped = 0
        self.media_found = 0
        self.media_downloaded = 0
        self.media_failed = 0
        
    def print_summary(self):
        """Print download statistics summary"""
        logging.info("\nDownload Statistics:")
        logging.info(f"Threads processed: {self.threads_processed}")
        logging.info(f"Threads skipped: {self.threads_skipped}")
        logging.info(f"Media files found: {self.media_found}")
        logging.info(f"Successfully downloaded: {self.media_downloaded}")
        logging.info(f"Failed downloads: {self.media_failed}")

# Global stats object
stats = DownloadStats()

# Set up directories and files
CHAN_DIR = "data/chan"
IMAGES_DIR = os.path.join(CHAN_DIR, "images")
VIDEOS_DIR = os.path.join(CHAN_DIR, "videos")
GIFS_DIR = os.path.join(CHAN_DIR, "gifs")
DOWNLOADED_THREADS_FILE = os.path.join(CHAN_DIR, "downloaded_threads.json")

# 4chan API endpoints
THREAD_API = "https://a.4cdn.org/{board}/thread/{thread_id}.json"
CATALOG_API = "https://a.4cdn.org/{board}/catalog.json"
IMAGE_BASE_URL = "https://i.4cdn.org/{board}/{tim}{ext}"

# Rate limiting settings
REQUEST_DELAY = 1.0    # Delay between requests in seconds
MAX_RETRIES = 3       # Maximum number of retries for 429 errors
RETRY_DELAY = 5       # Initial delay between retries in seconds
BACKOFF_FACTOR = 2    # Multiply delay by this factor after each retry

def setup_directories():
    """Create necessary directories if they don't exist"""
    os.makedirs(CHAN_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(VIDEOS_DIR, exist_ok=True)
    os.makedirs(GIFS_DIR, exist_ok=True)
    
    # Initialize downloaded threads tracking file if it doesn't exist
    if not os.path.exists(DOWNLOADED_THREADS_FILE):
        with open(DOWNLOADED_THREADS_FILE, 'w') as f:
            json.dump({}, f)

def load_downloaded_threads():
    """Load the set of downloaded thread IDs for each board"""
    try:
        with open(DOWNLOADED_THREADS_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading downloaded threads: {e}")
        return {}

def mark_thread_downloaded(board, thread_id):
    """Mark a thread as downloaded"""
    try:
        downloaded = load_downloaded_threads()
        if board not in downloaded:
            downloaded[board] = []
        if thread_id not in downloaded[board]:
            downloaded[board].append(thread_id)
            with open(DOWNLOADED_THREADS_FILE, 'w') as f:
                json.dump(downloaded, f, indent=2)
    except Exception as e:
        logging.error(f"Error marking thread as downloaded: {e}")

def is_thread_downloaded(board, thread_id):
    """Check if a thread has been downloaded"""
    downloaded = load_downloaded_threads()
    return board in downloaded and thread_id in downloaded[board]

def get_thread_title(thread_data):
    """Extract and clean thread title from thread data"""
    if not thread_data or 'posts' not in thread_data or not thread_data['posts']:
        return None
        
    op_post = thread_data['posts'][0]
    
    # Try to get subject first, fall back to comment snippet
    title = op_post.get('sub', '')
    if not title and 'com' in op_post:
        # Remove HTML tags and get first few words
        title = re.sub(r'<[^>]+>', '', op_post['com'])
        title = ' '.join(title.split()[:5])  # First 5 words
        
    if not title:
        return None
        
    # Clean the title for use as directory name
    title = re.sub(r'[<>:"/\\|?*]', '', title)  # Remove invalid chars
    title = title.strip()[:50]  # Limit length
    return title if title else None

def get_thread_directory(board, thread_id, thread_data, media_type):
    """Get the directory path for a specific thread's media"""
    # Get base directory for media type
    if media_type == 'image':
        base_dir = IMAGES_DIR
    elif media_type == 'video':
        base_dir = VIDEOS_DIR
    else:  # gif
        base_dir = GIFS_DIR
        
    # Get thread title
    thread_title = get_thread_title(thread_data)
    if thread_title:
        dir_name = f"{thread_id}_{thread_title}"
    else:
        dir_name = str(thread_id)
        
    # Create thread-specific directory
    thread_dir = os.path.join(base_dir, board, dir_name)
    os.makedirs(thread_dir, exist_ok=True)
    
    return thread_dir

def get_file_extension(filename):
    """Return file extension in lowercase"""
    return os.path.splitext(filename)[1].lower()

def get_file_type(ext):
    """
    Determine file type based on extension
    
    Returns:
        'image' for .jpg, .jpeg, .png, .webp
        'gif' for .gif
        'video' for .webm, .mp4
        'unknown' for others
    """
    if ext in [".jpg", ".jpeg", ".png", ".webp"]:
        return "image"
    elif ext == ".gif":
        return "gif"
    elif ext in [".webm", ".mp4"]:
        return "video"
    else:
        return "unknown"

def make_request(url, stream=False):
    """Make a request with retry logic for rate limits"""
    current_delay = RETRY_DELAY
    
    for attempt in range(MAX_RETRIES):
        try:
            if attempt > 0:
                logging.debug(f"Retry attempt {attempt + 1}")
            
            response = requests.get(url, stream=stream)
            
            if response.status_code == 429:  # Too Many Requests
                retry_after = int(response.headers.get('Retry-After', current_delay))
                retry_after = max(retry_after, current_delay)  # Use the longer delay
                logging.warning(f"Rate limited. Waiting {retry_after} seconds...")
                time.sleep(retry_after)
                current_delay *= BACKOFF_FACTOR  # Increase delay for next attempt
                continue
                
            response.raise_for_status()
            
            # Add delay between requests
            time.sleep(REQUEST_DELAY)
            
            return response
            
        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                logging.warning(f"Request failed: {e}. Retrying in {current_delay} seconds...")
                time.sleep(current_delay)
                current_delay *= BACKOFF_FACTOR
            else:
                logging.error(f"Failed to download after {MAX_RETRIES} attempts: {e}")
                return None

def download_file(url, output_path):
    """Download a file from URL to specified path"""
    try:
        response = make_request(url, stream=True)
        if not response:
            return False
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        logging.info(f"Downloaded: {url} -> {output_path}")
        return True
    except Exception as e:
        logging.error(f"Failed to download {url}: {e}")
        return False

def get_thread_data(board, thread_id):
    """Get JSON data for a specific thread"""
    url = THREAD_API.format(board=board, thread_id=thread_id)
    try:
        response = make_request(url)
        return response.json()
    except Exception as e:
        logging.error(f"Failed to get thread data: {e}")
        return None

def get_board_catalog(board):
    """Get complete catalog of threads for a board"""
    url = CATALOG_API.format(board=board)
    try:
        response = make_request(url)
        return response.json()
    except Exception as e:
        logging.error(f"Failed to get board catalog: {e}")
        return None

def process_post(post, board):
    """Extract media information from a post"""
    if 'filename' not in post or 'tim' not in post or 'ext' not in post:
        return None
        
    media_url = IMAGE_BASE_URL.format(
        board=board,
        tim=post['tim'],
        ext=post['ext']
    )
    
    return {
        'url': media_url,
        'filename': f"{post['filename']}{post['ext']}",
        'file_type': get_file_type(post['ext']),
        'original_name': post['filename'],
        'timestamp': post['tim']
    }

def download_thread_media(board, thread_id, skip_existing=False):
    """Download all media from a specific thread"""
    # Check if thread was already downloaded
    if skip_existing and is_thread_downloaded(board, thread_id):
        logging.info(f"Skipping previously downloaded thread: /{board}/{thread_id}")
        stats.threads_skipped += 1
        return
        
    stats.threads_processed += 1
        
    thread_data = get_thread_data(board, thread_id)
    if not thread_data or 'posts' not in thread_data:
        logging.error("Failed to get thread data or no posts found")
        return
        
    media_items = []
    for post in thread_data['posts']:
        media = process_post(post, board)
        if media:
            media_items.append(media)
    
    if not media_items:
        logging.info("No media found in thread")
        return
        
    stats.media_found += len(media_items)
    success = True
    
    for item in media_items:
        if item['file_type'] not in ['image', 'video', 'gif']:
            logging.debug(f"Skipping unsupported file type: {item['file_type']}")
            continue
            
        # Get thread-specific directory
        thread_dir = get_thread_directory(board, thread_id, thread_data, item['file_type'])
            
        # Create filename with timestamp
        filename = f"{item['timestamp']}{get_file_extension(item['filename'])}"
        output_path = os.path.join(thread_dir, filename)
        
        if download_file(item['url'], output_path):
            stats.media_downloaded += 1
        else:
            stats.media_failed += 1
            success = False
            
    if success:
        mark_thread_downloaded(board, thread_id)

def download_board_media(board, limit=None, skip_existing=False):
    """
    Download media from threads in a board
    
    Args:
        board: Board name
        limit: Maximum number of threads to download (None for all threads)
        skip_existing: Skip threads that have been previously downloaded
    """
    catalog = get_board_catalog(board)
    if not catalog:
        logging.error("Failed to get board catalog")
        return
        
    thread_ids = []
    for page in catalog:
        for thread in page.get('threads', []):
            thread_ids.append(thread['no'])
            
    if limit:
        thread_ids = thread_ids[:limit]
            
    logging.info(f"Downloading {len(thread_ids)} threads from /{board}/")
    
    for thread_id in thread_ids:
        logging.info(f"Processing thread {thread_id}")
        download_thread_media(board, thread_id, skip_existing)

def main():
    parser = argparse.ArgumentParser(
        description="Download media from 4chan threads or entire boards"
    )
    parser.add_argument(
        "board",
        help="Board name (e.g., 'w', 'wg', etc.)"
    )
    parser.add_argument(
        "--thread",
        type=str,
        help="Specific thread ID to download from"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of threads to download from board"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip threads that have been previously downloaded"
    )
    parser.add_argument(
        "--log-level",
        choices=['debug', 'info', 'warning', 'error'],
        default='info',
        help="Set logging level (default: info)"
    )
    
    args = parser.parse_args()
    
    # Setup logging and directories
    setup_logging(args.log_level)
    setup_directories()
    
    if args.thread:
        # Download specific thread
        logging.info(f"Downloading media from /{args.board}/{args.thread}")
        download_thread_media(args.board, args.thread, args.skip_existing)
    else:
        # Download board threads
        if args.limit:
            logging.info(f"Downloading up to {args.limit} threads from /{args.board}/")
        else:
            logging.info(f"Downloading all threads from /{args.board}/")
        download_board_media(args.board, args.limit, args.skip_existing)

if __name__ == "__main__":
    try:
        main()
    finally:
        # Always print stats summary at the end
        stats.print_summary()
