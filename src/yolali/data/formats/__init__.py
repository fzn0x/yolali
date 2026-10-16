"""Label readers. Each returns ``list[LabelRecord]`` without touching pixels."""

from yolali.data.formats.coco_json import read_coco_json
from yolali.data.formats.record import IMAGE_SUFFIXES, LabelRecord, image_size, list_images
from yolali.data.formats.yolo_txt import read_yolo_txt

__all__ = [
    "IMAGE_SUFFIXES",
    "LabelRecord",
    "image_size",
    "list_images",
    "read_coco_json",
    "read_yolo_txt",
]
