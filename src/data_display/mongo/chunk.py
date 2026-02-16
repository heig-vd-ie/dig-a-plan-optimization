import json
import math
import os
from pathlib import Path
from typing import Dict, Any

from tqdm import tqdm
from konfig import settings


def get_file_size_mb(file_path: Path) -> float:
    """Get file size in MB."""
    return file_path.stat().st_size / (1024 * 1024)


def is_sddp_response_file(data: Dict[str, Any]) -> bool:
    """Check if the JSON data is an SDDP response file."""
    return (
        isinstance(data, dict)
        and "simulations" in data
        and "objectives" in data
        and "out_of_sample_simulations" in data
        and "out_of_sample_objectives" in data
        and isinstance(data["simulations"], list)
        and isinstance(data["objectives"], list)
        and isinstance(data["out_of_sample_simulations"], list)
        and isinstance(data["out_of_sample_objectives"], list)
    )


def chunk_sddp_file(
    file_path: Path, chunk_size: int = 500, max_file_size_mb: int = 15
) -> bool:
    """
    Chunk a large SDDP file into smaller files in the same directory.

    Args:
        file_path: Path to the SDDP JSON file
        chunk_size: Number of simulations per chunk
        max_file_size_mb: File size threshold for chunking

    Returns:
        True if file was chunked, False if not needed
    """
    file_size_mb = get_file_size_mb(file_path)

    if file_size_mb <= max_file_size_mb:
        return False

    print(f"\033[32mChunking large file: {file_path} ({file_size_mb:.1f} MB)\033[0m")

    try:
        with file_path.open() as f:
            data = json.load(f)
    except (json.JSONDecodeError, MemoryError) as e:
        print(f"\033[31mError loading {file_path}: {e}\033[0m")
        return False

    if not is_sddp_response_file(data):
        print(
            f"\033[31mFile {file_path} is not an SDDP response file, skipping chunking\033[0m"
        )
        return False

    simulations = data["simulations"]
    objectives = data["objectives"]
    out_of_sample_simulations = data["out_of_sample_simulations"]
    out_of_sample_objectives = data["out_of_sample_objectives"]
    total_simulations = len(simulations)
    total_chunks = math.ceil(total_simulations / chunk_size)

    base_name = file_path.stem
    chunk_dir = file_path.parent / f"{base_name}_chunks"
    chunk_dir.mkdir(exist_ok=True)

    print(f"Creating {total_chunks} chunks with {chunk_size} simulations each")

    # Create chunks
    for i in tqdm(
        range(0, total_simulations, chunk_size),
        total=total_chunks,
        desc="Creating chunks",
    ):
        end_idx = min(i + chunk_size, total_simulations)
        chunk_index = i // chunk_size

        chunk_data = {
            "simulations": simulations[i:end_idx],
            "objectives": objectives[i:end_idx],
            "out_of_sample_simulations": out_of_sample_simulations[i:end_idx],
            "out_of_sample_objectives": out_of_sample_objectives[i:end_idx],
        }

        chunk_file = chunk_dir / f"{base_name}_chunk_{chunk_index:04d}.json"
        with chunk_file.open("w") as f:
            json.dump(chunk_data, f)

    # Create metadata file
    metadata = {
        "original_file": str(file_path),
        "total_chunks": total_chunks,
        "chunk_size": chunk_size,
        "total_simulations": total_simulations,
        "original_file_size_mb": file_size_mb,
    }

    metadata_file = chunk_dir / f"{base_name}_metadata.json"
    with metadata_file.open("w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\033[32m  Created metadata: {metadata_file}\033[0m")
    print(f"\033[32mSuccessfully chunked {file_path} into {total_chunks} files\033[0m")

    return True


def chunk_directory(directory: Path, chunk_size: int = 500, max_file_size_mb: int = 15):
    """Chunk all large SDDP files in a directory."""
    if not directory.exists():
        print(f"\033[31mDirectory {directory} does not exist\033[0m")
        return

    print(f"\033[32mScanning {directory} for large SDDP files...\033[0m")

    chunked_count = 0
    for json_file in directory.rglob("*.json"):
        # Skip already chunked files
        if "_chunk_" in json_file.name or json_file.name.endswith("_metadata.json"):
            continue

        if chunk_sddp_file(json_file, chunk_size, max_file_size_mb):
            chunked_count += 1

    print(f"\033[32mChunked {chunked_count} large files\033[0m")


def chunk_files():
    """Main function to chunk files in settings.cache.outputs_expansion directory."""
    base_dir = Path(settings.cache.outputs_expansion)

    if not base_dir.exists():
        print(f"\033[31mDirectory {base_dir} does not exist\033[0m")
        exit(1)
    # Default settings
    chunk_size = int(os.getenv("CHUNK_SIZE", "500"))
    max_file_size_mb = int(os.getenv("MAX_FILE_SIZE_MB", "15"))

    print(
        f"\033[32mChunking files larger than {max_file_size_mb}MB with {chunk_size} simulations per chunk\033[0m"
    )

    chunk_directory(base_dir, chunk_size, max_file_size_mb)
