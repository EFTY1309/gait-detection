"""
download_clips.py
-----------------
Download walking clips from YouTube and organize them into the enrollment/query
folder structure expected by enroll.py and recognize.py.

Usage
-----
  # Enroll clips — assign a person name to each URL
  python download_clips.py enroll --person Alice --url "https://youtu.be/XXXX"
  python download_clips.py enroll --person Bob   --url "https://youtu.be/YYYY"

  # Query clip — a video with multiple people to test recognition on
  python download_clips.py query --url "https://youtu.be/ZZZZ" --name test_scene

Requirements
------------
  pip install yt-dlp

Tips for finding good clips
----------------------------
  - Search YouTube: "person walking side view full body"
  - Search YouTube: "walking gait analysis"
  - Pexels free stock: https://www.pexels.com/search/videos/walking/
    (Pexels videos are free to download directly from their site)
  - For enrollment: pick videos where ONE person is clearly visible, full body,
    walking for at least 3-5 seconds.
"""

import argparse
import os
import subprocess
import sys


def check_ytdlp():
    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        print("yt-dlp not installed. Running: pip install yt-dlp")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "yt-dlp"])


def download(url: str, output_path: str) -> None:
    import yt_dlp

    ydl_opts = {
        "format": "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": output_path,
        "merge_output_format": "mp4",
        "quiet": False,
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


def main():
    parser = argparse.ArgumentParser(
        description="Download walking videos for gait recognition."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- enroll subcommand
    enroll_parser = subparsers.add_parser(
        "enroll", help="Download an enrollment clip for a specific person."
    )
    enroll_parser.add_argument("--person", required=True, help="Person's name (e.g. Alice)")
    enroll_parser.add_argument("--url", required=True, help="YouTube URL")
    enroll_parser.add_argument(
        "--clip-num", type=int, default=1,
        help="Clip number if enrolling multiple clips per person (default: 1)"
    )

    # -- query subcommand
    query_parser = subparsers.add_parser(
        "query", help="Download a query video to run recognition on."
    )
    query_parser.add_argument("--url", required=True, help="YouTube URL")
    query_parser.add_argument("--name", default="query", help="Output filename (no extension)")

    args = parser.parse_args()
    check_ytdlp()

    if args.command == "enroll":
        person_dir = os.path.join("data", "enrollment_videos", args.person)
        os.makedirs(person_dir, exist_ok=True)
        out_path = os.path.join(person_dir, f"clip{args.clip_num}.mp4")
        print(f"\nDownloading enrollment clip for '{args.person}' → {out_path}")
        download(args.url, out_path)
        print(f"\nDone. Now run:")
        print(f"  python enroll.py --video \"{out_path}\" --person {args.person}")

    elif args.command == "query":
        os.makedirs(os.path.join("data", "query_videos"), exist_ok=True)
        out_path = os.path.join("data", "query_videos", f"{args.name}.mp4")
        print(f"\nDownloading query video → {out_path}")
        download(args.url, out_path)
        print(f"\nDone. Now run:")
        print(f"  python recognize.py --video \"{out_path}\"")


if __name__ == "__main__":
    main()
