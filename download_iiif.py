import sys
import os
import argparse
from tqdm import tqdm
from Modules import IIIFDownloader



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workid", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=False, default="Data")
    args = parser.parse_args()

    work_id = args.workid
    output_dir = args.output_dir

    if not os.path.isdir(output_dir):
        sys.exit(f"'{output_dir}' is not a valid directory.")

    iiif_manifest = f"https://iiifpres.bdrc.io/collection/wio:bdr:M{work_id}::bdr:{work_id}"

    iiif_downloader = IIIFDownloader(output_dir=output_dir)
    iiif_downloader.download(iiif_manifest, file_limit=0)