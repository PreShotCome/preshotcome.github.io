import json
import logging
import os
import re
import subprocess
import sys
import time
from multiprocessing import Pool
from urllib.parse import urljoin

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("xhamster_downloader.log"), logging.StreamHandler()],
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/100.0.4844.84 Safari/537.36"
    ),
    "Referer": "https://xhamster.com/",
}


def get_video_id(url):
    """Extracts the video ID from the given xHamster video URL."""
    match = re.search(r"/videos/.+?-([\w\d]+)(?:\?|$)", url)
    if match:
        video_id = match.group(1)
        logging.info(f"Extracted Video ID: {video_id}")
        return video_id
    logging.error("Failed to extract video ID from URL.")
    return None


def fetch_page_content(url):
    """Fetches the HTML content of the given URL."""
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        logging.error(f"Failed to fetch {url}: {e}")
        return None


def extract_json_data(html_content):
    """Extracts JSON data from the HTML content of the embed page."""
    patterns = [
        r"window\.initials\s*=\s*({.*?});\s*</script>",
        r"window\.xplayerSettings\s*=\s*({.*?});\s*</script>",
    ]
    for pattern in patterns:
        match = re.search(pattern, html_content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                logging.error("Failed to parse JSON data.")
    logging.error("No valid JSON data found on the page.")
    return None


def get_playlists(video_id):
    """Retrieves H264 and AV1 playlists for a given video ID."""
    embed_url = f"https://xhamster.com/embed/{video_id}"
    logging.info(f"Fetching embed page: {embed_url}")
    content = fetch_page_content(embed_url)
    if not content:
        return None

    json_data = extract_json_data(content)
    if not json_data:
        return None

    sources = json_data.get("xplayerSettings", {}).get("sources", {})
    standard_section = sources.get("standard", {})

    playlists = {}
    for codec in ["h264", "av1"]:
        codec_data = standard_section.get(codec)
        if isinstance(codec_data, list) and codec_data:
            playlists[codec] = codec_data[0].get("url")
        elif isinstance(codec_data, dict):
            playlists[codec] = codec_data.get("url")

    return playlists if playlists else None


def get_best_quality_stream(master_url, min_res=144, max_res=2160):
    """Selects the best quality stream URL from the master playlist."""
    content = fetch_page_content(master_url)
    if not content:
        return None

    streams = []
    lines = content.strip().split("\n")
    for i in range(len(lines)):
        if lines[i].startswith("#EXT-X-STREAM-INF"):
            match = re.search(r"RESOLUTION=(\d+)x(\d+)", lines[i])
            if match and i + 1 < len(lines):
                streams.append(
                    {
                        "height": int(match.group(2)),
                        "url": urljoin(master_url, lines[i + 1].strip()),
                    }
                )

    if not streams:
        logging.error("No streams found.")
        return None

    best_stream = max(
        (s for s in streams if min_res <= s["height"] <= max_res),
        key=lambda s: s["height"],
        default=None,
    )
    return best_stream["url"] if best_stream else None


def download_and_process_h264(stream_url, title, destination):
    """Downloads and processes an H264 video stream using FFmpeg."""
    logging.info(f"Downloading H264 stream: {stream_url}")
    os.makedirs(destination, exist_ok=True)
    output_path = os.path.join(destination, f"{title}.mp4")

    command = ["ffmpeg", "-y", "-i", stream_url, "-c", "copy", output_path]
    subprocess.run(command, check=True)
    logging.info(f"Video saved as: {output_path}")


def main():
    """Main function to download xHamster videos based on URL input."""
    if len(sys.argv) < 2:
        print("Usage: python download_xhamster.py <video_url>")
        return

    video_url = sys.argv[1]
    destination_folder = os.path.join(os.getcwd(), "data", "videos")

    video_id = get_video_id(video_url)
    if not video_id:
        return

    playlists = get_playlists(video_id)
    if not playlists:
        return

    title = video_id  # Simplified title assignment

    if "h264" in playlists:
        logging.info("Downloading H264 video...")
        best_stream = get_best_quality_stream(playlists["h264"])
        if best_stream:
            download_and_process_h264(best_stream, title, destination_folder)
        else:
            logging.error("No valid H264 stream found.")
    else:
        logging.warning("H264 stream not available.")


if __name__ == "__main__":
    main()
