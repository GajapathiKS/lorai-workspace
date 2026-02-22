"""LorAI Semantic File System — ChromaDB-backed file indexer.

Watches ~/Documents, ~/Desktop, ~/Downloads for file changes
and indexes them into ChromaDB for semantic search.
"""

from __future__ import annotations

import hashlib
import os
import time

WATCH_DIRS = [
    os.path.expanduser("~/Documents"),
    os.path.expanduser("~/Desktop"),
    os.path.expanduser("~/Downloads"),
]

TEXT_EXTENSIONS = {
    ".txt", ".md", ".py", ".js", ".ts", ".json", ".yaml", ".yml",
    ".toml", ".cfg", ".ini", ".sh", ".bash", ".html", ".css",
    ".csv", ".log", ".xml", ".rst", ".tex",
}


def get_chroma_collection():
    """Get or create the ChromaDB collection for file indexing."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path="/data/vectors")
        return client.get_or_create_collection(
            name="lorai_files",
            metadata={"hnsw:space": "cosine"},
        )
    except Exception as e:
        print(f"ChromaDB not available: {e}")
        return None


def index_file(collection, filepath: str) -> bool:
    """Index a single text file into ChromaDB."""
    if collection is None:
        return False

    ext = os.path.splitext(filepath)[1].lower()
    if ext not in TEXT_EXTENSIONS:
        return False

    try:
        with open(filepath, "r", errors="replace") as f:
            content = f.read(50000)  # limit to 50KB
    except (OSError, PermissionError):
        return False

    if not content.strip():
        return False

    file_id = hashlib.md5(filepath.encode()).hexdigest()

    try:
        collection.upsert(
            ids=[file_id],
            documents=[content],
            metadatas=[{
                "path": filepath,
                "filename": os.path.basename(filepath),
                "extension": ext,
                "size": os.path.getsize(filepath),
                "modified": os.path.getmtime(filepath),
            }],
        )
        return True
    except Exception as e:
        print(f"Failed to index {filepath}: {e}")
        return False


def scan_and_index(collection) -> int:
    """Scan watch directories and index all text files."""
    count = 0
    for watch_dir in WATCH_DIRS:
        if not os.path.isdir(watch_dir):
            continue
        for root, _, files in os.walk(watch_dir):
            for fname in files:
                filepath = os.path.join(root, fname)
                if index_file(collection, filepath):
                    count += 1
    return count


def watch_loop():
    """Main watch loop — periodically re-scan and index files."""
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler

        collection = get_chroma_collection()

        class FileHandler(FileSystemEventHandler):
            def on_created(self, event):
                if not event.is_directory:
                    index_file(collection, event.src_path)

            def on_modified(self, event):
                if not event.is_directory:
                    index_file(collection, event.src_path)

        observer = Observer()
        for watch_dir in WATCH_DIRS:
            if os.path.isdir(watch_dir):
                observer.schedule(FileHandler(), watch_dir, recursive=True)

        # Initial scan
        count = scan_and_index(collection)
        print(f"LorAI FS: Indexed {count} files from watched directories.")

        observer.start()
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()

    except ImportError:
        # Fallback: poll-based scanning
        print("LorAI FS: watchdog not available, using polling mode.")
        collection = get_chroma_collection()
        while True:
            scan_and_index(collection)
            time.sleep(300)  # re-scan every 5 minutes


if __name__ == "__main__":
    watch_loop()
