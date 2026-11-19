"""Rebuild the 64 selected photographs from the GQA scene graphs and images."""
import argparse
import json
import shutil
from pathlib import Path

from cloud_prep.measurement_v12 import p1_data, p1_fetch, p1_prepare
from cloud_prep.portable_data import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, help="Directory of original GQA <image_id>.jpg files. Otherwise fetch selected JPEGs from the official image archive.")
    parser.add_argument("--output", type=Path, default=p1_data.ROOT)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    config = json.loads(Path(__file__).with_name("photo_selection.json").read_text())
    if output.exists() and any(output.iterdir()):
        parser.error("--output must be a new or empty directory")
    output.mkdir(parents=True, exist_ok=True)
    (output / "sources").mkdir()
    p1_data.make_index(output, excluded_ids=config["excluded_source_ids"])
    for image in config["selected_photos"]:
        image_id = image["image_id"]
        dest = output / "sources" / (image_id + ".jpg")
        if args.images:
            shutil.copyfile(args.images / dest.name, dest)
        else:
            result = p1_fetch.fetch_member(image_id, image["archive_member"], output)
            if result["status"] != "downloaded":
                raise RuntimeError(result)
        if sha256_file(dest) != image["sha256"]:
            raise ValueError(f"Original JPEG differs for {image_id}")
    p1_data.pixel_candidates(output)
    selected = [{key: row[key] for key in ("image_id", "query_descriptions", "note")} for row in config["selected_photos"]]
    p1_data.dump(output / "visual_review.json", {"status": "complete", "selected": selected})
    print(json.dumps(p1_prepare.build(output), indent=2))


if __name__ == "__main__":
    main()
