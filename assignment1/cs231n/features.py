from __future__ import print_function
from past.builtins import xrange

import matplotlib
import numpy as np
from scipy.ndimage import uniform_filter


def extract_features(imgs, feature_fns, verbose=False):
  """
  Given pixel data for images and several feature functions that can operate on
  single images, apply all feature functions to all images, concatenating the
  feature vectors for each image and storing the features for all images in
  a single matrix.

  Inputs:
  - imgs: N x H X W X C array of pixel data for N images.
  - feature_fns: List of k feature functions. The ith feature function should
    take as input an H x W x D array and return a (one-dimensional) array of
    length F_i.
  - verbose: Boolean; if true, print progress.

  Returns:
  An array of shape (N, F_1 + ... + F_k) where each column is the concatenation
  of all features for a single image.
  """
  num_images = imgs.shape[0]
  if num_images == 0:
    return np.array([])

  # Use the first image to determine feature dimensions
  feature_dims = []
  first_image_features = []
  for feature_fn in feature_fns:
    feats = feature_fn(imgs[0].squeeze())
    #print("feats.shape is :",feats.shape)
    assert len(feats.shape) == 1, 'Feature functions must be one-dimensional'
    feature_dims.append(feats.size)
    first_image_features.append(feats)

  # Now that we know the dimensions of the features, we can allocate a single
  # big array to store all features as columns.
  total_feature_dim = sum(feature_dims)
  imgs_features = np.zeros((num_images, total_feature_dim))
  imgs_features[0] = np.hstack(first_image_features).T

  # Extract features for the rest of the images.
  for i in xrange(1, num_images):
    idx = 0
    for feature_fn, feature_dim in zip(feature_fns, feature_dims):
      next_idx = idx + feature_dim
      imgs_features[i, idx:next_idx] = feature_fn(imgs[i].squeeze())
      idx = next_idx
    if verbose and i % 1000 == 0:
      print('Done extracting features for %d / %d images' % (i, num_images))

  return imgs_features


def rgb2gray(rgb):
  """Convert RGB image to grayscale

    Parameters:
      rgb : RGB image

    Returns:
      gray : grayscale image
  
  """
  return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])


def hog_feature(im):
  """Compute Histogram of Gradient (HOG) feature for an image
  
       Modified from skimage.feature.hog
       http://pydoc.net/Python/scikits-image/0.4.2/skimage.feature.hog
     
     Reference:
       Histograms of Oriented Gradients for Human Detection
       Navneet Dalal and Bill Triggs, CVPR 2005
     
    Parameters:
      im : an input grayscale or rgb image
      
    Returns:
      feat: Histogram of Gradient (HOG) feature
    
  """
  
  # convert rgb to grayscale if needed
  if im.ndim == 3:
    image = rgb2gray(im)
  else:
    image = np.at_least_2d(im)

  sx, sy = image.shape # image size
  orientations = 9 # number of gradient bins
  cx, cy = (8, 8) # pixels per cell

  gx = np.zeros(image.shape)
  gy = np.zeros(image.shape)
  gx[:, :-1] = np.diff(image, n=1, axis=1) # compute gradient on x-direction
  gy[:-1, :] = np.diff(image, n=1, axis=0) # compute gradient on y-direction
  grad_mag = np.sqrt(gx ** 2 + gy ** 2) # gradient magnitude
  grad_ori = np.arctan2(gy, (gx + 1e-15)) * (180 / np.pi) + 90 # gradient orientation

  n_cellsx = int(np.floor(sx / cx))  # number of cells in x
  n_cellsy = int(np.floor(sy / cy))  # number of cells in y
  # compute orientations integral images
  orientation_histogram = np.zeros((n_cellsx, n_cellsy, orientations))
  for i in xrange(orientations):
    # create new integral image for this orientation
    # isolate orientations in this range
    temp_ori = np.where(grad_ori < 180 / orientations * (i + 1),
                        grad_ori, 0)
    temp_ori = np.where(grad_ori >= 180 / orientations * i,
                        temp_ori, 0)
    # select magnitudes for those orientations
    cond2 = temp_ori > 0
    temp_mag = np.where(cond2, grad_mag, 0)
    # print("cx/2 is :",cx/2)
    # print("cy/2 is :",cy/2)

    orientation_histogram[:,:,i] = uniform_filter(temp_mag, size=(cx, cy))[int(cx/2)::cx, int(cy/2)::cy].T
  
  return orientation_histogram.ravel()


def color_histogram_hsv(im, nbin=10, xmin=0, xmax=255, normalized=True):
  """
  Compute color histogram for an image using hue.

  Inputs:
  - im: H x W x C array of pixel data for an RGB image.
  - nbin: Number of histogram bins. (default: 10)
  - xmin: Minimum pixel value (default: 0)
  - xmax: Maximum pixel value (default: 255)
  - normalized: Whether to normalize the histogram (default: True)

  Returns:
    1D vector of length nbin giving the color histogram over the hue of the
    input image.
  """
  ndim = im.ndim
  bins = np.linspace(xmin, xmax, nbin+1)
  hsv = matplotlib.colors.rgb_to_hsv(im/xmax) * xmax
  imhist, bin_edges = np.histogram(hsv[:,:,0], bins=bins, density=normalized)
  imhist = imhist * np.diff(bin_edges)

  #print("shape of the imhist is : ",imhist.shape)
  # return histogram
  return imhist


#pass

def eigValPct(eigVals,percentage):
    sortArray=np.sort(eigVals) #使用numpy中的sort()对特征值按照从小到大排序
    sortArray=sortArray[-1::-1] #特征值从大到小排序
    arraySum=sum(sortArray) #数据全部的方差arraySum
    tempSum=0
    num=0
    for i in sortArray:
        tempSum+=i
        num+=1
        if(tempSum>=arraySum*percentage):
            return num

def pca_my(im):
  # convert rgb to grayscale if needed
  #print("in the pca function")
  if im.ndim == 3:
    image = rgb2gray(im)
  else:
    image = np.at_least_2d(im)

  sx, sy = image.shape  # image size
  #print("the size of the image is :",sx)
  #print("the size of the image is :",sy)
  topNfeat=90
  meanVals = np.mean(image, axis=0)
    # print 'meanVals', meanVals

    # 每个向量同时都减去 均值
  meanRemoved = image - meanVals
    # print 'meanRemoved=', meanRemoved


  covMat = np.cov(meanRemoved, rowvar=0)

  # eigVals为特征值， eigVects为特征向量
  eigVals, eigVects = np.linalg.eig(np.mat(covMat))
  eigValInd = np.argsort(eigVals)
    # print 'eigValInd1=', eigValInd


  k=eigValPct(eigVals,0.9) #要达到方差的百分比percentage，需要前k个向量
  eigValInd=np.argsort(eigVals)  #对特征值eigVals从小到大排序
  eigValInd=eigValInd[:-(k+1):-1] #从排好序的特征值，从后往前取k个，这样就实现了特征值的从大到小排列
  redEigVects=eigVects[:,eigValInd]   #返回排序后特征值对应的特征向量redEigVects（主成分）
  lowDDataMat=meanRemoved*redEigVects #将原始数据投影到主成分上得到新的低维数据lowDDataMat
  reconMat=(lowDDataMat*redEigVects.T)+meanVals   #得到重构数据reconMat


  #   # -1表示倒序，返回topN的特征值[-1 到 -(topNfeat+1) 但是不包括-(topNfeat+1)本身的倒叙]
  # eigValInd = eigValInd[:-(topNfeat+1):-1]
  #   # print 'eigValInd2=', eigValInd
  #   # 重组 eigVects 最大到最小
  # redEigVects = eigVects[:, eigValInd]
  #   # print 'redEigVects=', redEigVects.T
  #   # 将数据转换到新空间
  #   # print "---", shape(meanRemoved), shape(redEigVects)
  # lowDDataMat = meanRemoved * redEigVects
  # reconMat = (lowDDataMat * redEigVects.T) + meanVals
  #reconMat_one=reconMat.reshape(reconMat.shape[0]*reconMat.shape[1])
  reconMat_one=reconMat.reshape(1024,1)
  #np.resize(a, (2, 3))
  reconMat_test=np.resize(reconMat_one,(1024))
  #print("shape of the reconMat_test is :",reconMat_test.shape)

  #print("shape of the reconMat_one is :",reconMat_one.shape)

 # print("shape of the changed ")

  return reconMat_test

