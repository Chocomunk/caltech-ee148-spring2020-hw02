import os
import numpy as np
import json
from PIL import Image

from util import correlate, get_boxes, load_filters, printProgressBar, gray_dilation, min_box_size

def compute_convolution(I, T, stride=None):
    '''
    This function takes an image <I> and a template <T> (both numpy arrays) 
    and returns a heatmap where each grid represents the output produced by 
    convolution at each location. You can add optional parameters (e.g. stride, 
    window_size, padding) to create additional functionality. 
    '''
    (n_rows,n_cols,n_channels) = np.shape(I)

    '''
    BEGIN YOUR CODE
    '''
    # heatmap = np.random.random((n_rows, n_cols))
    if not stride:
        stride = 1
    heatmap = correlate(I, T, stride)
    '''
    END YOUR CODE
    '''

    return heatmap


def predict_boxes(heatmap):
    '''
    This function takes heatmap and returns the bounding boxes and associated
    confidence scores.
    '''

    output = []

    '''
    BEGIN YOUR CODE
    '''

    output = get_boxes(heatmap)
    
    '''
    As an example, here's code that generates between 1 and 5 random boxes
    of fixed size and returns the results in the proper format.
    '''

    # box_height = 8
    # box_width = 6

    # num_boxes = np.random.randint(1,5)

    # for i in range(num_boxes):
    #     (n_rows,n_cols,n_channels) = np.shape(I)

    #     tl_row = np.random.randint(n_rows - box_height)
    #     tl_col = np.random.randint(n_cols - box_width)
    #     br_row = tl_row + box_height
    #     br_col = tl_col + box_width

    #     score = np.random.random()

    #     output.append([tl_row,tl_col,br_row,br_col, score])

    '''
    END YOUR CODE
    '''

    return output


def detect_red_light_mf(I):
    '''
    This function takes a numpy array <I> and returns a list <output>.
    The length of <output> is the number of bounding boxes predicted for <I>. 
    Each entry of <output> is a list <[row_TL,col_TL,row_BR,col_BR,score]>. 
    The first four entries are four integers specifying a bounding box 
    (the row and column index of the top left corner and the row and column 
    index of the bottom right corner).
    <score> is a confidence score ranging from 0 to 1. 

    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''

    '''
    BEGIN YOUR CODE
    '''
    # template_height = 8
    # template_width = 6

    # You may use multiple stages and combine the results
    # T = np.random.random((template_height, template_width))
    heatmaps = np.zeros((len(templates), I.shape[0], I.shape[1]))
    for i in range(len(templates)):
        ### Weakened Algorithm
        heatmaps[i] = compute_convolution(I, templates[i], stride=2)
        ### Best Algorithm
        # conv = compute_convolution(I, T, stride=1)

    max_map = np.max(heatmaps, axis=0)
    output = []
    for i in range(len(templates)):
        boxes = predict_boxes((heatmaps[i] > thresholds[i]) * max_map)
        output += min_box_size(boxes, 
                            templates[i].shape[0], templates[i].shape[1],
                            I.shape[0], I.shape[1])

    '''
    END YOUR CODE
    '''

    for i in range(len(output)):
        assert len(output[i]) == 5
        assert (output[i][4] >= 0.0) and (output[i][4] <= 1.0)

    return output


if __name__ == '__main__':
    # Note that you are not allowed to use test data for training.
    # set the path to the downloaded data:
    data_path = 'data/RedLights2011_Medium'

    # load splits: 
    split_path = 'data/hw02_splits'
    file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
    file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

    # Red-light template setup
    filters, compound_template = load_filters('red-lights/balance')

    ### Weakened Algorithm
    templates = [filters[1]]
    thresholds = [0.6]

    ### Best Algorithm
    # templates = filters
    # thresholds = [0.73, 0.765, 0.655]
    # templates = [filters[2]]      # Turns out just 3rd picture is best :(
    # thresholds = [0.655]

    # set a path for saving predictions:
    preds_path = 'data/hw02_preds'
    os.makedirs(preds_path, exist_ok=True) # create directory if needed

    # Set this parameter to True when you're done with algorithm development:
    done_tweaking = False

    # Since we seed the randomness, we can avoid executing on the training set
    # again for some extra speed
    train_model = True

    '''
    Make predictions on the training set.
    '''
    if train_model:
        print("Running predictions on training set")
        preds_train = {}
        n = len(file_names_train)
        printProgressBar(0, n, prefix='Progress:', suffix='Complete', length=50)
        for i in range(len(file_names_train)):

            # read image using PIL:
            I = Image.open(os.path.join(data_path,file_names_train[i]))

            # convert to numpy array:
            I = np.asarray(I)

            preds_train[file_names_train[i]] = detect_red_light_mf(I)

            # Intermediate saves just incase
            if (i+1) % 50 == 0:
                # print("Writing to preds_train_{}.json".format(i+1))
                with open(os.path.join(preds_path,'preds_train_{}.json'.format(i+1)),'w') as f:
                    json.dump(preds_train,f)

            printProgressBar(i, n, prefix='Progress:', suffix='Complete', length=50)

        # save preds (overwrites any previous predictions!)
        print("Writing to preds_train.json")
        with open(os.path.join(preds_path,'preds_train.json'),'w') as f:
            json.dump(preds_train,f)

    '''
    Make predictions on the test set. 
    '''
    if done_tweaking:
        print("Running predictions on test set")
        preds_test = {}
        n = len(file_names_test)
        printProgressBar(0, n, prefix='Progress:', suffix='Complete', length=50)
        for i in range(len(file_names_test)):

            # read image using PIL:
            I = Image.open(os.path.join(data_path,file_names_test[i]))

            # convert to numpy array:
            I = np.asarray(I)

            preds_test[file_names_test[i]] = detect_red_light_mf(I)

            # Intermediate saves just incase
            if (i+1) % 50 == 0:
                # print("Writing to preds_test_{}.json".format(i+1))
                with open(os.path.join(preds_path,'preds_test_{}.json'.format(i+1)),'w') as f:
                    json.dump(preds_test,f)

            printProgressBar(i, n, prefix='Progress:', suffix='Complete', length=50)

        # save preds (overwrites any previous predictions!)
        print("Writing to preds_test.json")
        with open(os.path.join(preds_path,'preds_test.json'),'w') as f:
            json.dump(preds_test,f)
