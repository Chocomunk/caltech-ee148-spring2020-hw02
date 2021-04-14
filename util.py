import os
import numpy as np

from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from PIL import Image


def norm(img):
    """ Normalize an image down to a 'unit vector' """
    n = np.linalg.norm(img)
    if n == 0:
        return img
    return img.astype(np.float32) / n


def rgb2gray(img):
    """ Convert RGB image to grayscale values """
    # return np.dot(img, [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    return np.dot(img, [0.2125, 0.7154, 0.0721]).astype(np.uint8)
    # return np.mean(img, axis=2).astype(np.uint8)


def gray2rgb(img):
    """ Expand grayscale image into RGB (still gray) """
    if len(img.shape) == 2:
        return np.repeat(img[:,:,np.newaxis], 3, axis=2)
    return img
    


def rgb2hsv(img):
    """ Use matplotlib to convert rgb to hsv (TA allowed) """
    return (rgb_to_hsv(img.astype(np.float32) / 255.) * 255).astype(np.uint8)


def hsv2rgb(img):
    """ Use matplotlib to convert hsv to rgb (TA allowed) """
    return (hsv_to_rgb(img.astype(np.float32) / 255.) * 255).astype(np.uint8)


# from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html
def histogram_equalization(img, bins=128, clip_limit=None):
    """ Use histogram equalization to balance contrast over the entire image """
    hist, hist_bins = np.histogram(img.flatten(), bins)
    if clip_limit is not None:      # Clip and redistribute (simplified)
        clip_mask = (hist < clip_limit)
        distr = np.sum(hist * (1 - clip_mask) - clip_limit) / np.sum(clip_mask)
        hist = np.clip(hist + distr * clip_mask, 0, clip_limit)
    cdf = hist.cumsum()
    cdf = 255 * cdf / cdf[-1]
    equalized = np.interp(img.flatten(), hist_bins[:-1], cdf)
    return equalized.reshape(img.shape)


# THIS IMPLEMENTATION IS INCORRECT, DO NOT USE
def CLAHE(img, clip_limit=2.0, tile_size=(8,8), bins=128):
    """ Balance contrast locally over an image using tiling approximation """
    n, m = img.shape[:2]
    u, v = tile_size
    output = np.zeros(img.shape)
    for r in range(max(1, (n-1) // u + 1)):         # Round up integer div
        for c in range(max(1, (m-1) // v + 1)):
            end_r = min(n, (r+1)*u)
            end_c = min(m, (c+1)*v)
            output[r*u:end_r, c*v:end_c] = histogram_equalization(
                img[r*u:end_r, c*u:end_c], bins=bins, clip_limit=clip_limit)
    return output


def binary_dilation(img, iterations=1):
    """ Dilates a mask with a square structuring element """
    output = np.copy(img)
    n, m = img.shape[:2]

    for _ in range(iterations):         # Move left
        output[:,:m-1] |= output[:,1:]
    for _ in range(iterations):         # Move right
        output[:,1:] |= output[:,:m-1]
    for _ in range(iterations):         # Move up
        output[:n-1] |= output[1:]
    for _ in range(iterations):         # Move down
        output[1:] |= output[:n-1]

    return output


def binary_erosion(img, iterations=1):
    """ Erodes a mask with a square structuring element """
    output = np.copy(img)
    n, m = img.shape[:2]

    for _ in range(iterations):         # Move left
        output[:,:m-1] &= output[:,1:]
    for _ in range(iterations):         # Move right
        output[:,1:] &= output[:,:m-1]
    for _ in range(iterations):         # Move up
        output[:n-1] &= output[1:]
    for _ in range(iterations):         # Move down
        output[1:] &= output[:n-1]

    return output


def gray_dilation(img, iterations=1):
    """ Dilates a grayscale image with a square structuring element """
    if len(img.shape) == 3:
        output = np.max(img, axis=2)
    else:
        output = np.copy(img)
    n, m = img.shape[:2]

    for _ in range(iterations):         # Move left
        np.maximum(output[:,:m-1], output[:,1:], output[:,:m-1])
    for _ in range(iterations):         # Move right
        np.maximum(output[:,1:], output[:,:m-1], output[:,1:])
    for _ in range(iterations):         # Move up
        np.maximum(output[:n-1], output[1:], output[:n-1])
    for _ in range(iterations):         # Move down
        np.maximum(output[1:], output[:n-1], output[1:])

    return gray2rgb(output)


def gray_erosion(img, iterations=1):
    """ Erodes a grayscale image with a square structuring element """
    if len(img.shape) == 3:
        output = np.max(img, axis=2)
    else:
        output = np.copy(img)
    n, m = img.shape[:2]

    for _ in range(iterations):         # Move left
        np.minimum(output[:,:m-1], output[:,1:], output[:,:m-1])
    for _ in range(iterations):         # Move right
        np.minimum(output[:,1:], output[:,:m-1], output[:,1:])
    for _ in range(iterations):         # Move up
        np.minimum(output[:n-1], output[1:], output[:n-1])
    for _ in range(iterations):         # Move down
        np.minimum(output[1:], output[:n-1], output[1:])

    return gray2rgb(output)


def gray_opening(img, size=1):
    """ Computes the opening operation on a grayscale image """
    return gray_dilation(gray_erosion(img, iterations=size), iterations=size)


def gray_closing(img, size=1):
    """ Computes the closing operation on a grayscale image """
    return gray_erosion(gray_dilation(img, iterations=size), iterations=size)


def white_tophat(img, size=1):
    """ Applies a white-tophat transform to an image """
    return img - gray_opening(img, size=size)


def black_tophat(img, size=1):
    """ Applies a black-tophat transform to an image """
    return gray_closing(img, size=size) - img


def correlate(A, B, step=1):
    """ Correlates image B over image A. Assumes B is normalized """
    u, v = B.shape[:2]
    n, m = A.shape[:2]

    # Padding the input
    p1 = u // 2
    p2 = v // 2
    padded_img = np.pad(A, [(p1, p1), (p2, p2), (0, 0)], mode='constant')

    output = np.zeros((n,m))     # Output is same size as input
    for r in range(0, n, step):
        for c in range(0, m, step):
            window = norm(padded_img[r:r+u, c:c+v])
            output[r:r+step,c:c+step] = np.vdot(window, B)

    return output


def conv2d(img, filter):
    """ Convolves a grayscale image with a filter """
    k1, k2 = filter.shape[:2]
    n, m = img.shape[:2]

    if not k1 & k2 & 1:
        raise ValueError("Filter should have odd dimensions")

    # Padding the input
    p1 = k1 // 2
    p2 = k2 // 2
    padded_img = np.pad(img, [(p1, p1), (p2, p2)], mode='constant')

    output = np.zeros(img.shape)     # Output is same size as input
    for r in range(n):
        for c in range(m):
            window = padded_img[r:r+k1, c:c+k2]
            output[r,c] = np.sum(filter * window, axis=(0,1))

    return output


def get_boxes(mask, radius=3, mode="square"):
    """ Extracts bounding boxes from a grayscale mask 

    'mode' can be wither "square" or "circular" for how neighbors are selected.
    The effect of this parameter is more apparent at the edges
    """
    # Generate neighbors first
    neighbors = []
    if mode == "square":
        neighbors = [(r,c) for r in range(-radius+1,radius) 
                           for c in range(-radius+1,radius)]
        # neighbors = [(-radius, 0), (radius, 0), (0, radius), (0, -radius)]
    elif mode == "circular":
        neighbors = [(r,c) for r in range(-radius+1,radius) 
                           for c in range(-radius+1,radius)
                           if r*r + c*r <= radius*radius]
    else:
        raise ValueError("Unrecognized neighbor mode")
    neighbors = np.array(neighbors, dtype=np.int16)

    # BFS to find all objects
    n, m = mask.shape[:2]
    not_visited = (mask > 0)

    # All pixels in the image + edges + corners
    queue = np.zeros((mask.size, 2), dtype=np.int16)
    i = 0                               # End of queue

    boxes = []
    y1 = n; x1 = m; y2 = 0; x2 = 0      # Initialize bounding box
    conf = 0.                           # Initialize confidence
    num_pixels = 0
    x = 0; y = 0                        # For finding the bounding box

    while(not_visited.any()):
        # Find a non-zero element as the starting point
        queue[0] = np.argwhere(not_visited)[0]
        not_visited[queue[0][0], queue[0][1]] = False
        i = 1
        y1 = n; x1 = m; y2 = 0; x2 = 0      # Re-initialize bounding box
        conf = 0.                           # Re-initialize confidence
        num_pixels = 0
        while i > 0:
            i -= 1
            y, x = queue[i]

            # Compute bounding box and confidence intermediates
            y1 = min(y1, y); x1 = min(x1, x)
            y2 = max(y2, y); x2 = max(x2, x)
            conf += mask[y, x]
            num_pixels += 1

            # Populate queue with neighbors (if in image bounds and new)
            for n_y, n_x in (queue[i] + neighbors):
                if (0 <= n_x < m) and (0 <= n_y < n) and not_visited[n_y, n_x]:
                    queue[i] = n_y, n_x
                    not_visited[n_y, n_x] = False   # Stop future propagation
                    i += 1

        # Save bounding box of this object
        boxes.append([int(y1), int(x1), int(y2), int(x2), conf / num_pixels])

    return boxes


def min_box_size(boxes, height, width, n, m):
    """ Expands every bounding box to have the minimum size constraint 
    
    Arguments:
        width   - Minimum box width
        height  - Minimum box heigh
        n       - Number of rows in the input image
        m       - Number of cols in the input image
    """
    output = []
    for t, l, b, r, c in boxes:
        if (b-t) < height:
            padding = (height - b + t) // 2
            b = min(n-1, b + padding)
            t = max(0, t - padding)
        if (r-l) < width:
            padding = (width - r + l) // 2
            r = min(m-1, r + padding)
            l = max(0, l - padding)
        # if 2 * (r-l) < (b-t):       # Filter out shapes that aren't "tall" enough
        output.append([t, l, b, r, c])
    return output


def draw_boxes(img, bounding_boxes):
    """ Finds and draws red-light bounding boxes, returns the new image """
    I = np.copy(img)
    # Top, Left, Bottom, Right coords
    for t, l, b, r, c in bounding_boxes:
        c1 = c * 90
        c2 = c * 255
        # Clear red and green (add value for brightness)
        I[t:b,l,0:2] = c1      # left wall
        I[t:b,r,0:2] = c1      # right wall
        I[t,l:r,0:2] = c1      # top wall
        I[b,l:r,0:2] = c1      # bottom wall

        # Color in blue
        I[t:b,l,2] = c2      # left wall
        I[t:b,r,2] = c2      # right wall
        I[t,l:r,2] = c2      # top wall
        I[b,l:r,2] = c2      # bottom wall
    return I


def load_filters(lights_path):
    orig_filters = []
    filters = []
    for f_name in os.listdir(lights_path):
        filt_img = np.asarray(Image.open(os.path.join(lights_path,f_name)))
        # filt_img = histogram_equalization(filt_img, clip_limit=2)
        orig_filters.append(filt_img.astype(np.uint8))
        filters.append(norm(filt_img))

    # Generate compound img
    max_width = max([x.shape[1] for x in orig_filters])
    compound_filter = np.concatenate(
        [np.pad(x, [(0,10), (max_width - x.shape[1], 0), (0,0)], mode='constant')
        for x in orig_filters], 0)

    return filters, compound_filter


# https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()