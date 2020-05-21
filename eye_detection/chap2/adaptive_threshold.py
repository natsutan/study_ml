import os
from PIL import Image
import numpy as np

C = 12
img_fname = 'spinner_small.jpg'


def mean_c(im, c=10):
    # 7x7固定
    N = 3
    FS = 7
    im_gray = im.convert('L')

    im_arr = np.asarray(im_gray)
    h, w = im_arr.shape
    dst = np.zeros((h-(N+1), w-(N+1)), dtype=np.uint8)
    dst_h, dst_w = dst.shape

    # 境界処理はしない。
    for y in range(dst_h):
        for x in range(dst_w):
            neighbors = im_arr[y:y+FS,x:x+FS]
            i = im_arr[y+N][x+N]
            thresh = np.mean(neighbors) - c

            if np.isnan(thresh):
                continue

            if i < thresh:
                dst[y][x] = 0
            else:
                dst[y][x] = 255


    new_im = Image.fromarray(dst, mode='L')

    return new_im

def main():
    im = Image.open(os.path.join('img', img_fname))
    gt = mean_c(im, C)
    gt.save(os.path.join('out', 'ada_mean_c.png'))


if __name__ == '__main__':
    main()


