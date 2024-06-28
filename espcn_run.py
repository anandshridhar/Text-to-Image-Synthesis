import cv2
from models.espcn import ESPCN
import os
from pathlib import Path


for image in os.listdir('results'):
	img = cv2.imread(os.path.join('results', image))

	
#img=cv2.imread('results/img.jpg')
	sr = cv2.dnn_superres.DnnSuperResImpl_create()
	path = "/groups/ldbrown/t1capstone/kaggle/ESPCN/ESPCN_x4.pb"
	sr.readModel(path)
	sr.setModel("espcn", 4) # set the model by passing the value and the upsampling ratio
	result = sr.upsample(img) # upscale the input image
	cv2.imwrite(os.path.join('espcn_results',image), result)
