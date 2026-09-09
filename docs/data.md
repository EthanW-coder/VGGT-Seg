# Data Preparation

All commands are run from the repository root. Dataset depth maps and poses are used only to construct point-level labels and evaluation coordinates. The model forward pass receives RGB images only.

## ScanNet v2 and ScanNet200

Request ScanNet v2 data from the [official ScanNet site](http://www.scan-net.org/) and use the [official download tools](https://github.com/ScanNet/ScanNet). ScanNet200 uses the same scans and the official ScanNet200 labels distributed with the benchmark.

Extract the `.sens` sequences:

```bash
python tools/data/extract_scannet_frames.py \
  --scans-root /path/to/scans \
  --output-root /path/to/scannet_frames
```

Prepare ScanNet v2:

```bash
python tools/data/prepare_scannet.py \
  --benchmark scannet \
  --scans-root /path/to/scans \
  --frames-root /path/to/scannet_frames \
  --label-map /path/to/scannetv2-labels.combined.tsv \
  --train-split /path/to/scannetv2_train.txt \
  --val-split /path/to/scannetv2_val.txt \
  --output-root data/scannet
```

For ScanNet200, change `--benchmark` to `scannet200` and `--output-root` to `data/scannet200`.

## AI2THOR 3D

Download the [preprocessed AI2THOR 3D frames](https://drive.google.com/uc?id=1H9fiGZg8ILSOhfssRxF1rXOVKyfFwepa). Use the train and validation scene lists included with the data.

```bash
python tools/data/prepare_ai2thor.py \
  --frames-root /path/to/ai2thor_frames_512 \
  --train-split /path/to/ai2thor_train.txt \
  --val-split /path/to/ai2thor_val.txt \
  --output-root data/ai2thor
```

## Matterport3D

Download the [preprocessed Matterport3D frames](https://drive.google.com/uc?id=1mWU8jxrAlxsci7ste07qUo6S895kDt-f) and the accompanying processed point clouds. Prepare either taxonomy with:

```bash
python tools/data/prepare_matterport3d.py \
  --taxonomy 21 \
  --frames-root /path/to/matterport_frames \
  --point-root /path/to/matterport_points \
  --train-split /path/to/m3d_train.txt \
  --val-split /path/to/m3d_val.txt \
  --output-root data/matterport21
```

Use `--taxonomy 160` and `data/matterport160` for the 160-class setting.

