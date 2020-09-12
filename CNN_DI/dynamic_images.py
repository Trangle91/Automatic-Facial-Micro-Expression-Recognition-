
import numpy as np
import cv2
from inspect import signature

#images are a list of images or a 4-dimensional array of size: H x W x D x N
def compute_approximate_dynamic_images(images):
	if len(images) == 0:
		di = []

	if type(images) is np.ndarray:
		imagesA = np.zeros((1,len(images)))
		for i in range(1,len(images)):
			if not isinstance(i, str):
				print("images must be an array of images or cell of image names")
			imagesA[i] = cv2.imread(i)
		images = np.concat(imagesA,imagesA.shape[3])

	N = images.shape[3]
	singled_images = images.astype('float32')
	one_arrays = np.ones((1,N))
	di = vl_nnarpooltemporal(singled_images, one_arrays)


def vl_nnarpooltemporal(X,ids,dzdy):
	sz = X.shape()

	#getting the number of args
	sig = signature(vl_nnarpooltemporal)
	params = sig.parameters
	nargin = len(params)

	#not sure if logical should be coded here. Original code: logical(nargin < 3) -> returns a
	#numeric value, but forward is a boolean value here (look at its usage below), "nargin<3" itself
	#returns a boolean -> I don't think we need to convert its output into a int number.
	forward = nargin<3

	if len(ids) != X.shape[3]:
		print('Error: ids dimension does not match with X!')

	nVideos = np.amax(ids)

	if forward:
		Y = np.zeros((sz[:3],nVideos), 'like', X)
	else:
		Y = np.zeros((sz), 'like', X)

	for v in range(1, nVideos):
		indv = np.nonzero(ids[v] == v)
		if len(indv) == 0:
			print('Error: No frames in video %d',v)
		N = len(indv)
		fw = np.zeros((1,N))
		if N==1:
			fw = 1
		else:
			for i in range(1,N):

				# not sure 
				fw[i] = sum((2* [i:N] - N - 1) / [i:N])

		if forward:
			single_fw_ = fw.astype('float32')
			new_shape_ = (np.ones((1,4)).shape[3] = len(indv))
			reshaped_ = np.reshape(single_fw_, new_shape)
			product_ = np.multiply(X[:,:,:,indv], reshaped_)
			Y[:,:,:,v] = sum(product_,4)

		else:
			new_shape_ = (np.ones((1,4)).shape[3] = len(indv))
			remated_ = np.tile(dzdy[:,:,:,v], new_shape_)
			reshaped_ = np.reshape(fw, new_shape_)
			Y[:,:,:,indv] = np.multiply(remated_, reshaped_)


