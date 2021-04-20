# CS 148 Homework 2

## Purpose

Design an algorithm to detect red lights in images using convolutions.

## Usage

Run `run_predictions.py` to generate the bounding-box predictions at `data/hw02_preds`. Note, intermediate predictions will be generated every 50 images for safety.

```
python run_predictions.py
```

Next, run `eval_detector.py` to compare the generated BBoxes with ground-truth BBoxes provided by student labellers. This will generate Precision-Recall curves for training and testing data (depending on variables set) in the `data/` directory.

```
python eval_detector.py
```
