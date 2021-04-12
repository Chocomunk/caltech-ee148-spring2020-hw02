import os
import numpy as np
import matplotlib.pyplot as plt

from skimage.exposure import equalize_adapthist
from PIL import Image

from util import load_filters, get_boxes, draw_boxes, binary_dilation, gray2rgb, correlate, gray_dilation, min_box_size


# Thresholds
# heat_thresh = 0.695
heat_thresh = 0.655
# heat_thresh = 0.73
# heat_thresh = 0.745

CMAP = plt.cm.get_cmap('tab20')


def get_image(path, fname):
    I = np.asarray(Image.open(os.path.join(path,fname)))
    r = I.shape[0]
    return I[:2*r//3]


def gen_image(I):
    global recorrelate, corr
    if recorrelate:
        corr = np.zeros(I.shape[:2], dtype=np.float32)
        for filt in filters:
            corr = np.maximum(corr, correlate(I, filt, step=1))
        print(np.max(corr))
        recorrelate = False

    # dilate = gray_dilation(corr, iterations=5)[:,:,0]
    dilate = corr
    map_out = CMAP(dilate)[:,:,:3] * 255
    # output = binary_dilation((corr > heat_thresh), iterations=5)
    output = (dilate > heat_thresh) * corr
    boxes = get_boxes(output)
    boxes = min_box_size(boxes, 
                         filter_all[2].shape[0], filter_all[2].shape[1],
                         I.shape[0], I.shape[1])
    I = draw_boxes(I, boxes)

    left = np.concatenate((I, gray2rgb(output > 0) * I), 0)
    right = np.concatenate((gray2rgb(dilate * 255).astype(np.uint8), 
                            map_out.astype(np.uint8)), 0)

    height_diff = right.shape[0] - compound_filter.shape[0]
    filter_image = np.pad(compound_filter, [(0,height_diff), (0,0), (0,0)], mode='constant')
    return np.concatenate((left, right, filter_image), 1)


def press(event):
    global heat_thresh, i, I, looping, f_i, filters, recorrelate
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
            f_i = min(len(filter_all), f_i+1)
            filters = filter_sets[f_i]
            recorrelate = True
        if event.key == 'z':      # Previous filter
            f_i = max(0, f_i-1)
            filters = filter_sets[f_i]
            recorrelate = True
        elif event.key == 'w':
            heat_thresh = min(1, heat_thresh+0.1)
        elif event.key == 's':
            heat_thresh = max(0, heat_thresh-0.1)
        elif event.key == 'e':
            heat_thresh = min(1, heat_thresh+0.05)
        elif event.key == 'd':
            heat_thresh = max(0, heat_thresh-0.05)
        elif event.key == 'r':
            heat_thresh = min(1, heat_thresh+0.01)
        elif event.key == 'f':
            heat_thresh = max(0, heat_thresh-0.01)
        elif event.key == 't':
            heat_thresh = min(1, heat_thresh+0.005)
        elif event.key == 'g':
            heat_thresh = max(0, heat_thresh-0.005)

        # Update image
        im.set_array(gen_image(I))
        ax.set_ylabel('Filter Set: {0}'.format(f_i))
        ax.set_xlabel('Heatmap Threshold: {0}'.format(heat_thresh))
        fig.canvas.draw()


if __name__=="__main__":
    # Data images setup
    data_path = 'data/RedLights2011_Medium'
    # data_path = 'red-lights/balance'
    file_names = sorted(os.listdir(data_path)) 
    file_names = [f for f in file_names if '.jpg' in f] 
    n = len(file_names)

    # Red-light image setup
    f_i = 2
    filter_all, compound_filter = load_filters('red-lights/balance')
    filter_sets = [[filter_all[0]], [filter_all[1]], [filter_all[2]], filter_all]
    filters = filter_sets[f_i]

    # Loop sentinels
    i = 0
    looping = True
    recorrelate = True
    corr = None

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
