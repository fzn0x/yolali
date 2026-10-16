# datasets/

Datasets are downloaded here by `scripts/download_datasets.py` (phase 3) and are
git-ignored. Expected layout:

```
datasets/
├── coco128/        # 128-image COCO subset, YOLO txt labels (gate 4 smoke train)
├── coco/           # val2017 images + annotations/instances_val2017.json (gate 3)
└── voc/            # VOC2007+2012 converted to YOLO txt (gates 4 and 6)
```

Each loader also writes a `labels.cache` next to the image directory; delete it
to force a rebuild, or change any file and the content hash does it for you.
