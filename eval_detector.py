import os
import json
import numpy as np
import matplotlib.pyplot as plt


def compute_iou(box_1, box_2):
    '''
    This function takes a pair of bounding boxes and returns intersection-over-
    union (IoU) of two bounding boxes.
    '''
    t1, l1, b1, r1 = box_1
    t2, l2, b2, r2 = box_2

    # Get box of intersection
    top = None
    if t2 <= t1 <= b2:
        top = t1
    elif t1 <= t2 <= b1:
        top = t2

    left = None
    if l2 <= l1 <= r2:
        left = l1
    elif l1 <= l2 <= r1:
        left = l2
    
    bot = None
    if t2 <= b1 <= b2:
        bot = b1
    elif t1 <= b2 <= b1:
        bot = b2

    right = None
    if l2 <= r1 <= r2:
        right = r1
    elif l1 <= r2 <= r1:
        right = r2

    # If any walls are not defined, no intersection
    if top is None or left is None or bot is None or right is None:
        return 0

    # Box areas
    x1, y1 = r1 - l1, b1 - t1
    x2, y2 = r2 - l2, b2 - t2
    x, y = right - left, bot - top

    intersection = x * y
    union = (x1 * y1) + (x2 * y2) - intersection
    iou = intersection / float(union)
    
    assert (iou >= 0) and (iou <= 1.0)

    return iou


def compute_counts(preds, gts, iou_thr=0.5, conf_thr=0.5):
    '''
    This function takes a pair of dictionaries (with our JSON format; see ex.) 
    corresponding to predicted and ground truth bounding boxes for a collection
    of images and returns the number of true positives, false positives, and
    false negatives. 
    <preds> is a dictionary containing predicted bounding boxes and confidence
    scores for a collection of images.
    <gts> is a dictionary containing ground truth bounding boxes for a
    collection of images.
    '''
    TP = 0
    FP = 0
    FN = 0

    '''
    BEGIN YOUR CODE
    '''
    for pred_file, pred in preds.items():
        gt = gts[pred_file]
        M = 0
        n = 0

        # Each prediction only corresponds to 1 ground truth
        filled = [False for _ in range(len(pred))]
        for i in range(len(gt)):
            for j in range(len(pred)):

                # Only count if we are confident enough
                if not filled[j] and pred[j][4] >= conf_thr:
                    M += 1
                    iou = compute_iou(pred[j][:4], gt[i])
                    if iou >= iou_thr:
                        filled[j] = True
                        n += 1
                        break
        FP += M - n
        FN += len(gt) - n
        TP += n

    '''
    END YOUR CODE
    '''

    return TP, FP, FN


def compute_PR(preds, gts, iou_thresholds, conf_thresholds):
    """ Computes precision and recall of a dataset over the specified thresholds

    Arguments
        preds           - Model predictions
        gts             - Ground truth values
        iou_thresholds  - List of IoU thresholds to consider
        conf_thresholds - List of confidence thresholds to consider
    """
    tp = np.zeros((len(conf_thresholds), len(iou_thresholds)), dtype=np.int32)
    fp = np.zeros((len(conf_thresholds), len(iou_thresholds)), dtype=np.int32)
    fn = np.zeros((len(conf_thresholds), len(iou_thresholds)), dtype=np.int32)
    for j, iou_thr in enumerate(iou_thresholds):
        for i, conf_thr in enumerate(conf_thresholds):
            tp[i,j], fp[i,j], fn[i,j] = compute_counts(preds, gts, iou_thr=iou_thr, conf_thr=conf_thr)
    M = tp + fp
    N = tp + fn

    # Precision, Recall
    return np.divide(tp, M), np.divide(tp, N)


def plot_PR_curve(preds, gts, name):
    # conf_thrs = np.sort(np.array([box[4] for fname in preds for box in preds[fname]],dtype=float))
    conf_thrs = np.linspace(0.655, 0.99, 50)
    iou_thrs = [0.25, 0.5, 0.75]
    P, R = compute_PR(preds, gts, iou_thrs, conf_thrs)

    # Plot PR curves
    for i in range(len(iou_thrs)):
        plt.plot(R[:,i], P[:,i], label="IoU Thresh: {0}".format(iou_thrs[i]))

    plt.legend()
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve of {0}".format(name))


if __name__ == '__main__':
    # set a path for predictions and annotations:
    preds_path = 'data/hw02_preds'
    gts_path = 'data/hw02_annotations'

    # load splits:
    split_path = 'data/hw02_splits'
    file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
    file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

    # Set this parameter to True when you're done with algorithm development:
    done_tweaking = False

    '''
    Load training data. 
    '''
    with open(os.path.join(preds_path,'preds_train.json'),'r') as f:
        preds_train = json.load(f)
        
    with open(os.path.join(gts_path, 'annotations_train.json'),'r') as f:
        gts_train = json.load(f)

    if done_tweaking:
        
        '''
        Load test data.
        '''
        
        with open(os.path.join(preds_path,'preds_test.json'),'r') as f:
            preds_test = json.load(f)
            
        with open(os.path.join(gts_path, 'annotations_test.json'),'r') as f:
            gts_test = json.load(f)

    # Plot training set PR curves
    plot_PR_curve(preds_train, gts_train, "Training Set")
    plt.savefig('data/train_pr_curve')
    plt.show()
    if done_tweaking:
        plot_PR_curve(preds_test, gts_test, "Test Set")
        plt.savefig('data/test_pr_curve')
        plt.show()

    if done_tweaking:
        print('Code for plotting test set PR curves.')
