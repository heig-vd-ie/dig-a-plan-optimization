from config import settings
from general_function import download_from_switch


def download_load_allocation_data(FORCE_DOWNLOAD: bool = False):
    download_from_switch(
        local_folder_path=settings.LOAD_ALLOCATION_LOCAL_FOLDER,
        switch_folder_path="estavayer/load_allocation",
        switch_link=settings.SWITCH_LINK,
        switch_pass=settings.SWITCH_PASS,
        download_anyway=FORCE_DOWNLOAD,
    )


if __name__ == "__main__":
    download_load_allocation_data(FORCE_DOWNLOAD=False)
