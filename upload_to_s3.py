"""
S3 Multipart Upload with Real Per-Byte Progress Display
Uses boto3 s3transfer for actual upload progress
Supports uploading entire directories with multi-threading
"""

import boto3
import os
import math
import time
from boto3.s3.transfer import S3Transfer
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

load_dotenv()

# S3 client configuration
s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID', 'user_3CbWsXrurrIPUDzOrtCMXETfkbx'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY', 'rps_41GMS2JR1RCMX0QXUD28EONFA1WM5T4XF8O7QE4I1289u2'),
    region_name='eu-cz-1',
    endpoint_url='https://s3api-eu-cz-1.runpod.io'
)

BUCKET = '4qinybex2c'
LOCAL_DIR = r'/home/wlia0047/ar57/wenyu/result'
KEY_PREFIX = 'result/'
MAX_WORKERS = 32  # 并发上传线程数


def format_size(size_bytes):
    """Format bytes to human readable string"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"


class ProgressPrinter:
    def __init__(self, file_size):
        self.file_size = file_size
        self.bytes_uploaded = 0
        self.last_time = time.time()
        self.start_time = time.time()
        self.last_bytes = 0

    def __call__(self, bytes_amount):
        self.bytes_uploaded += bytes_amount
        now = time.time()

        # Update at least 10 times per second or on completion
        if now - self.last_time >= 0.1 or self.bytes_uploaded >= self.file_size:
            self._print()
            self.last_time = now
            self.last_bytes = self.bytes_uploaded

    def _print(self):
        now = time.time()
        elapsed = now - self.start_time
        speed = self.bytes_uploaded / elapsed if elapsed > 0 else 0

        progress = self.bytes_uploaded / self.file_size * 100
        bar_width = 50
        filled = min(int(bar_width * self.bytes_uploaded / self.file_size), bar_width)
        bar = '=' * filled + '-' * (bar_width - filled)

        eta = (self.file_size - self.bytes_uploaded) / speed if speed > 0 else 0
        if eta > 3600:
            eta_str = f"{eta/3600:.1f}h"
        elif eta > 60:
            eta_str = f"{eta/60:.1f}m"
        else:
            eta_str = f"{eta:.0f}s"

        # Current transfer speed (instantaneous)
        instant_speed = (self.bytes_uploaded - self.last_bytes) / max(now - self.last_time, 0.01)

        print(f"\r[{bar}] {progress:6.2f}% | {self.bytes_uploaded:>12} / {self.file_size} bytes | {format_size(speed):>10}/s | ETA: {eta_str:<8}", end='', flush=True)


def upload_file_with_progress(local_path, s3_key, file_size):
    """Upload single file using S3Transfer with real per-byte progress"""
    transfer = S3Transfer(s3)
    progress = ProgressPrinter(file_size)

    transfer.upload_file(
        local_path,
        BUCKET,
        s3_key,
        extra_args=None,
        callback=progress
    )


class UploadStats:
    def __init__(self, total_files, total_size):
        self.total_files = total_files
        self.total_size = total_size
        self.uploaded_size = 0
        self.completed_files = 0
        self.start_time = time.time()
        self.lock = Lock()

    def update(self, file_size):
        with self.lock:
            self.uploaded_size += file_size
            self.completed_files += 1
            elapsed = time.time() - self.start_time
            avg_speed = self.uploaded_size / elapsed if elapsed > 0 else 0
            return self.completed_files, self.total_files, self.uploaded_size, self.total_size, avg_speed


def upload_single_file(args):
    """Upload single file to S3"""
    local_path, s3_key, file_size, stats = args

    transfer = S3Transfer(s3)
    transfer.upload_file(local_path, BUCKET, s3_key)

    completed, total, uploaded, total_size, speed = stats.update(file_size)
    print(f"[{completed}/{total}] {os.path.basename(local_path):40s} | {format_size(speed):>10}/s | {completed*100//total}%")

    return local_path, s3_key


def upload_directory(local_dir, key_prefix, max_workers=8):
    """Upload entire directory to S3 with multi-threaded parallel uploads"""
    # Collect all files first
    files_to_upload = []
    total_size = 0

    for root, dirs, files in os.walk(local_dir):
        for filename in files:
            local_path = os.path.join(root, filename)
            relative_path = os.path.relpath(local_path, local_dir)
            s3_key = os.path.join(key_prefix, relative_path).replace(os.sep, '/')
            file_size = os.path.getsize(local_path)
            files_to_upload.append((local_path, s3_key, file_size))
            total_size += file_size

    stats = UploadStats(len(files_to_upload), total_size)

    print(f"Directory: {local_dir}")
    print(f"Files: {len(files_to_upload)}")
    print(f"Total Size: {format_size(total_size)}")
    print(f"Concurrency: {max_workers} threads")
    print()

    # Prepare args with stats
    args_list = [(local_path, s3_key, file_size, stats)
                 for local_path, s3_key, file_size in files_to_upload]

    print("=== Starting Multi-Threaded Upload ===")
    start_time = time.time()

    # Use ThreadPoolExecutor for parallel uploads
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(upload_single_file, args) for args in args_list]
        for future in as_completed(futures):
            future.result()  # Wait for completion

    elapsed = time.time() - start_time
    avg_speed = total_size / elapsed if elapsed > 0 else 0

    print(f"\n\n=== Upload Complete ===")
    print(f"Total: {format_size(total_size)} ({total_size} bytes)")
    print(f"Files: {len(files_to_upload)}")
    print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f}m)")
    print(f"Avg speed: {format_size(avg_speed)}/s")
    print(f"S3 Key Prefix: {key_prefix}")


if __name__ == '__main__':
    upload_directory(LOCAL_DIR, KEY_PREFIX)
