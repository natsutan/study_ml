import os
from PIL import Image

TRESHOLD = 80
img_fname = 'spinner.jpg'


def global_thresh(im, thresh=100):
    im_gray = im.convert('L')
    return im_gray.point(lambda x: 0 if x < thresh else 255)

def main():
    im = Image.open(os.path.join('img', img_fname))
    gt = global_thresh(im, TRESHOLD)
    gt.save(os.path.join('out', 'global_thresh.png'))


if __name__ == '__main__':
    main()


