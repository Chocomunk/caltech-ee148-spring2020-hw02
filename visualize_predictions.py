
import os
import json
import numpy as np
from PIL import Image

from util import draw_boxes, printProgressBar


if __name__=='__main__':
    # Directory for output visualizations
    out_path = 'data/visualizations'
    os.makedirs(out_path,exist_ok=True)     # create directory if needed 

    # Load input images
    data_path = 'data/RedLights2011_Medium'

    # Load predictions
    preds_path = 'data/hw02_preds'
    preds_file = 'preds_test_50.json'       # TODO: Change this as needed
    data = None
    with open(os.path.join(preds_path,preds_file),'r') as f:
        data = json.load(f)
    assert data is not None

    # Draw and save bounding boxes for each file
    i = 0
    n = len(data)
    printProgressBar(0, n, prefix='Progress:', suffix='Complete', length=50)
    for file_name, boxes in data.items():
        # Read image using PIL
        I = np.asarray(Image.open(os.path.join(data_path,file_name)))
        result = draw_boxes(I, boxes)

        # Save new image
        Image.fromarray(result).save(os.path.join(out_path,file_name))

        printProgressBar(i, n, prefix='Progress:', suffix='Complete', length=50)
        i += 1
        
