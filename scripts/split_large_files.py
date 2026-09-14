#!/usr/bin/env python3
import os
import glob
import subprocess

MAX_SIZE_BYTES = 90 * 1024 * 1024  # 90 MB

def find_large_files(root_dir="."):
    large_files = []
    ignored_dirs = {"venv", ".venv", ".git", "node_modules", ".next"}
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in ignored_dirs]
        for f in files:
            if f.endswith(".part_aa") or ".part_" in f:
                continue
            file_path = os.path.join(root, f)
            try:
                if os.path.islink(file_path):
                    continue
                size = os.path.getsize(file_path)
                if size > MAX_SIZE_BYTES:
                    large_files.append((file_path, size))
            except OSError:
                pass
    return large_files

def split_file(file_path, size):
    print(f"Splitting '{file_path}' ({size / (1024*1024):.2f} MB)...")
    prefix = file_path + ".part_"
    cmd = ["split", "-b", "90m", file_path, prefix]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error splitting {file_path}: {res.stderr}")
        return False
    print(f"Successfully split '{file_path}' into parts.")
    # Remove original large file to prevent git tracking & save disk space
    os.remove(file_path)
    return True

def main():
    root = os.path.abspath(".")
    large_files = find_large_files(root)
    print(f"Found {len(large_files)} files larger than 90MB.")
    for file_path, size in large_files:
        split_file(file_path, size)

if __name__ == "__main__":
    main()
