import os
import numpy as np
import matplotlib.pyplot as plt

from skimage.exposure import equalize_adapthist
from PIL import Image

from util import load_filters, get_boxes, draw_boxes, binary_dilation, gray2rgb, correlate, gray_dilation, min_box_size


# Thresholds
# heat_thresh = 0.695
# heat_thresh = 0.655
# heat_thresh = 0.73
thresh_ind = 0
heat_thresh = [0.73, 0.765, 0.655]

CMAP = plt.cm.get_cmap('tab20')


def get_image(path, fname):
    I = np.asarray(Image.open(os.path.join(path,fname)))
    r = I.shape[0]
    return I[:2*r//3]


def gen_image(I):
    global recorrelate, corr, heatmaps, map_out
    if recorrelate:
        heatmaps = np.zeros((len(filters_ind), I.shape[0], I.shape[1]), dtype=np.float32)
        for j, i in enumerate(filters_ind):
            heatmaps[j] = correlate(I, filters[i], step=2)
        recorrelate = False
        corr = np.max(heatmaps, axis=0)
        map_out = CMAP(corr)[:,:,:3] * 255
        print(np.max(corr))

    boxes = []
    output = np.zeros(I.shape[:2], dtype=np.float32)
    for j, i in enumerate(filters_ind):
        mask = (heatmaps[j] > heat_thresh[i]) * corr
        output = np.maximum(output, mask)
        mask_boxes = get_boxes(mask)
        boxes += min_box_size(mask_boxes,
                            filters[i].shape[0], filters[i].shape[1],
                            I.shape[0], I.shape[1])
    I = draw_boxes(I, boxes)

    left = np.concatenate((I, gray2rgb(output > 0) * I), 0)
    right = np.concatenate((gray2rgb(corr * 255).astype(np.uint8), 
                            map_out.astype(np.uint8)), 0)

    height_diff = right.shape[0] - compound_filter.shape[0]
    filter_image = np.pad(compound_filter, [(0,height_diff), (0,0), (0,0)], mode='constant')
    return np.concatenate((left, right, filter_image), 1)


def press(event):
    global heat_thresh, i, I, looping, f_i, filters_ind, recorrelate, thresh_ind
    # Control commands
    if event.key == 'q':        # Stop program completely
        looping = False
        plt.close()
    else:                       # Updates image after keypress
        if event.key == 'n':      # Next image
            i = min(n-1, i+1)
            I = get_image(data_path, file_names[i])
            recorrelate = True
        if event.key == 'b':      # Previous image
            i = max(0, i-1)
            I = get_image(data_path, file_names[i])
            recorrelate = True
        if event.key == 'x':      # Next filter
            f_i = min(len(filter_sets), f_i+1)
            filters_ind = filter_sets[f_i]
            recorrelate = True
        if event.key == 'z':      # Previous filter
            f_i = max(0, f_i-1)
            filters_ind = filter_sets[f_i]
            recorrelate = True

        # Select which threshold to edit
        elif event.key == 'u':
            thresh_ind = 0
        elif event.key == 'i':
            thresh_ind = 1
        elif event.key == 'o':
            thresh_ind = 2

        # Change threshhold value
        elif event.key == 'w':
            heat_thresh[thresh_ind] = min(1, heat_thresh[thresh_ind]+0.1)
        elif event.key == 's':
            heat_thresh[thresh_ind] = max(0, heat_thresh[thresh_ind]-0.1)
        elif event.key == 'e':
            heat_thresh[thresh_ind] = min(1, heat_thresh[thresh_ind]+0.05)
        elif event.key == 'd':
            heat_thresh[thresh_ind] = max(0, heat_thresh[thresh_ind]-0.05)
        elif event.key == 'r':
            heat_thresh[thresh_ind] = min(1, heat_thresh[thresh_ind]+0.01)
        elif event.key == 'f':
            heat_thresh[thresh_ind] = max(0, heat_thresh[thresh_ind]-0.01)
        elif event.key == 't':
            heat_thresh[thresh_ind] = min(1, heat_thresh[thresh_ind]+0.005)
        elif event.key == 'g':
            heat_thresh[thresh_ind] = max(0, heat_thresh[thresh_ind]-0.005)

        # Update image
        im.set_array(gen_image(I))
        ax.set_ylabel('Filter Set: {0}, Thresh Index {1}'.format(f_i, thresh_ind))
        ax.set_xlabel('Heatmap Thresholds: {0}, {1}, {2}'.format(*heat_thresh))
        fig.canvas.draw()


if __name__=="__main__":
    # Data images setup
    data_path = 'data/RedLights2011_Medium'
    # data_path = 'red-lights/balance'
    file_names = sorted(os.listdir(data_path)) 
    file_names = [f for f in file_names if '.jpg' in f] 
    n = len(file_names)

    # Red-light image setup
    f_i = 3
    filters, compound_filter = load_filters('red-lights/balance')
    # filter_sets = [[filter_all[0]], [filter_all[1]], [filter_all[2]], filter_all]
    filter_sets = [[0], [1], [2], [0, 1, 2]]
    filters_ind = filter_sets[f_i]

    # Loop sentinels
    i = 0
    looping = True
    recorrelate = True
    corr = None
    heatmaps = []

    # Figure management
    fig, ax = plt.subplots()
    I = get_image(data_path, file_names[i])
    im = ax.imshow(gen_image(I))
    ax.set_ylabel('Filter Set: {0}'.format(f_i))
    ax.set_xlabel('Heatmap Threshold: {0}'.format(heat_thresh))
    ax.set_title("Press 'q' to quit, 'n' for next image, 'b' for prev image")

    # Bind keypress controller
    fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
    fig.canvas.mpl_connect('key_press_event', press)

    # Main loop
    plt.show()
