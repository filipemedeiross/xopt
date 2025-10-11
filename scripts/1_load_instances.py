import os
from urllib import request

from .constants import INSTANCES_URL,  \
                       INSTANCES_FILES


def download_file(url, dest_path):
    try:
        print(f"Downloading {os.path.basename(dest_path)}...")
        request.urlretrieve(url, dest_path)
    except Exception as e:
        print(f"Failed to download {url}: {e}")


def loader():
    root_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
    instances_dir = os.path.join(root_dir, "instances")

    os.makedirs(instances_dir, exist_ok=True)

    for filename in INSTANCES_FILES:
        download_file(
            f"{INSTANCES_URL}{filename}",
            os.path.join(instances_dir, filename)
        )

    print()
    print("All p-median instances and best-known values downloaded successfully.")


if __name__ == "__main__":
    loader()
