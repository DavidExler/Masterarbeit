# Data Folder

### Expected outputs:
| Filename | Generating Tab | Explanation |
|----------|----------|----------|
| images.pkl | Segmentation | Python list of images (Z, X, Y, C) for faster read access |
| masks.pkl | Segmentation | Python list of segmentation masks (Z, X, Y) |
| label_store.json | Labeling App | Readable label data used for classificator training |
| pseudo_labels.json | Optimize | Readable label data used for semi supervised classificator pretraining |