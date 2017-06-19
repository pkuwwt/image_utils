# -*- coding: utf-8 -*-
"""
Apply the histogram of one image to another image, so that the hue is transferred

"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

def single_channel_hist(image, channel):
    """
    Calculate cumulative histgram of single image channel 
    return as a list of length 256
    """
    hist = cv2.calcHist([image], [channel], None, [256], [0.0, 255.0])
    histVals = [hist[i] for i in range(256)]
    s = sum(histVals)
    pdf = [v/s for v in histVals]
    for i in range(1, 256):
        pdf[i] = pdf[i-1] + pdf[i]
    return pdf

    
def cal_hist(image):
    return [single_channel_hist(image, i) for i in range(3)]

def cal_trans(src, dest):
    """
    Calculate the transfer function as a look-up table of length 256
    """
    table = list(range(256))
    for i in range(1,256):
        for j in range(1,256):
            if dest[i]>=src[j-1] and dest[i]<=src[j]:
                table[i] = j
                break
    table[255] = 255
    return table

def bgr2rgb(img):
    img2 = np.copy(img)
    img2[:,:,[0,2]] = img2[:,:,[2,0]]
    return img2

def histmatch(srcImg, destImg):
    """
    apply hist of srcImg to destImg
    return a new version of desgImg
    """
    srcHists = cal_hist(srcImg)
    destHists = cal_hist(destImg)
    tables = [cal_trans(srcHists[i], destHists[i]) for i in range(3)]
    nrows,ncols,nchannel = destImg.shape
    dest = np.zeros_like(destImg)
    for k in range(nchannel):
        for i in range(nrows):
            for j in range(ncols):
                dest[i,j,k] = tables[k][destImg[i,j,k]]
    return dest

def imshow(arr):
    arr2 = bgr2rgb(arr)
    plt.imshow(arr2)
    plt.show()
    
def imread(filename):
    return cv2.imread(filename)

def imwrite(filename, img):
    cv2.imwrite(filename, img)

def histmatch_file(srcFile, destFile, outFile):
    src = imread(srcFile)
    dest = imread(destFile)
    out = histmatch(src, dest)
    imwrite(outFile, out)
    
def test():
    lena = imread('lena.jpg')
    airplane = imread('airplane.jpg')
    out = histmatch(lena, airplane)
    #cv2.imwrite('airplane2.jpg', out)
    imshow(lena)
    imshow(airplane)
    imshow(out)

def test2():
    histmatch_file('lena.jpg', 'airplane.jpg', 'airplane2.jpg')
    histmatch_file('airplane.jpg', 'lena.jpg', 'lena2.jpg')
    
def main():
    if len(sys.argv)<4:
        print ("USAGE: {} ref.jpg input.jpg out.jpg".format(sys.argv[0]))
        return
    histmatch_file(sys.argv[1], sys.argv[2], sys.argv[3])
    
if __name__ == '__main__':
    main()