"""Download the datasets used by the verification gates into ``datasets/``.

    uv run scripts/download_datasets.py coco128      # 128 images, YOLO txt labels
    uv run scripts/download_datasets.py coco-val     # val2017 images + instances json
    uv run scripts/download_datasets.py voc          # VOC2007+2012 -> YOLO txt

Everything is idempotent: present files are not re-downloaded.
"""

from __future__ import annotations

import argparse
import shutil
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "datasets"

COCO128_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
COCO_VAL_IMAGES = "http://images.cocodataset.org/zips/val2017.zip"
COCO_VAL_ANN = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
VOC_URLS = {
    "VOCtrainval_06-Nov-2007.tar": "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",
    "VOCtest_06-Nov-2007.tar": "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar",
    "VOCtrainval_11-May-2012.tar": "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
}
VOC_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
    "train", "tvmonitor",
)  # fmt: skip


def download(url: str, dest: Path) -> Path:
    if dest.exists():
        print(f"have {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url} -> {dest}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
    return dest


def coco128() -> None:
    target = ROOT / "coco128"
    if (target / "images" / "train2017").is_dir():
        print(f"have {target}")
        return
    archive = download(COCO128_URL, ROOT / "coco128.zip")
    with zipfile.ZipFile(archive) as z:
        z.extractall(ROOT)
    archive.unlink()


def coco_val() -> None:
    target = ROOT / "coco"
    if not (target / "val2017").is_dir():
        archive = download(COCO_VAL_IMAGES, target / "val2017.zip")
        with zipfile.ZipFile(archive) as z:
            z.extractall(target)
        archive.unlink()
    if not (target / "annotations" / "instances_val2017.json").exists():
        archive = download(COCO_VAL_ANN, target / "annotations.zip")
        with zipfile.ZipFile(archive) as z:
            z.extract("annotations/instances_val2017.json", target)
        archive.unlink()
    print(f"have {target}")


def voc_xml_to_yolo(xml_path: Path, out_txt: Path) -> None:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    assert size is not None
    w = float(size.findtext("width") or 0)
    h = float(size.findtext("height") or 0)
    lines: list[str] = []
    for obj in root.iter("object"):
        name = obj.findtext("name")
        if name not in VOC_CLASSES or obj.findtext("difficult") == "1":
            continue
        bb = obj.find("bndbox")
        assert bb is not None
        x1, y1, x2, y2 = (float(bb.findtext(k) or 0) for k in ("xmin", "ymin", "xmax", "ymax"))
        cx, cy, bw, bh = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h
        lines.append(f"{VOC_CLASSES.index(name)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    out_txt.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def voc() -> None:
    """VOC07+12 trainval -> ``voc/images/train``, VOC07 test -> ``voc/images/val`` (YOLO layout)."""
    target = ROOT / "voc"
    if (target / "images" / "val").is_dir() and (target / "images" / "train").is_dir():
        print(f"have {target}")
        return
    raw = target / "raw"
    for name, url in VOC_URLS.items():
        archive = download(url, raw / name)
        marker = raw / f".{name}.extracted"
        if not marker.exists():
            print(f"extracting {name}")
            with tarfile.open(archive) as t:
                t.extractall(raw, filter="data")
            marker.touch()
    splits = {
        "train": [("VOC2007", "trainval"), ("VOC2012", "trainval")],
        "val": [("VOC2007", "test")],
    }
    for split, parts in splits.items():
        img_dir = target / "images" / split
        lbl_dir = target / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for year, subset in parts:
            base = raw / "VOCdevkit" / year
            ids = (base / "ImageSets" / "Main" / f"{subset}.txt").read_text().split()
            for image_id in ids:
                stem = f"{year}_{image_id}"
                dst = img_dir / f"{stem}.jpg"
                if not dst.exists():
                    shutil.copyfile(base / "JPEGImages" / f"{image_id}.jpg", dst)
                voc_xml_to_yolo(base / "Annotations" / f"{image_id}.xml", lbl_dir / f"{stem}.txt")
                n += 1
        print(f"{split}: {n} images")
    (target / "names.txt").write_text("\n".join(VOC_CLASSES) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("dataset", choices=["coco128", "coco-val", "voc"])
    args = parser.parse_args()
    {"coco128": coco128, "coco-val": coco_val, "voc": voc}[args.dataset]()


if __name__ == "__main__":
    main()
