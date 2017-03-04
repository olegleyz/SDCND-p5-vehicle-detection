from class_camera import *
from class_image import *
from skimage.feature import hog
from sklearn.utils import shuffle
from scipy.ndimage.measurements import label
from scipy.ndimage import maximum
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import time

def load_data(size='small',
					portion = 1,#2 - take each 2 image, 3 - take each 3rd image
					approach = '80/20',#'trtespl' - traintestsplit 
					cspace='YUV',
                    orient=9, 
                    pix_per_cell=8, 
                    cell_per_block=4, 
                    hog_channel=1,
                    feature_vec=True,
                    hist_feat=False,
                    spatial_feat=False):
	print ('Loading data')
	if size=='big':
		cars_url = '../datasets/vehicles/*/*.png'
		noncars_url = '../datasets/non-vehicles/*/*.png'
	else:
		cars_url = '../datasets/vehicles_smallset/*/*.jpeg'
		noncars_url = '../datasets/non-vehicles_smallset/*/*.jpeg'

	cars_urls = glob.glob(cars_url)
	# filtering images to take portion of them
	indices = [i for i in range(0,len(cars_urls),portion)]
	cars_urls = [cars_urls[i] for i in indices]
	
	noncars_urls = glob.glob(noncars_url)
	# filtering images to take portion of them
	indices = [i for i in range(0,len(noncars_urls),portion)]
	noncars_urls = [noncars_urls[i] for i in indices]
	
	print ('There are {} cars images and {} noncars images'.format(len(cars_urls), len(noncars_urls)))

	test_img = cv2.imread(cars_urls[0])[:,:,::-1]
	print ('Test image in the big dataset has shape {}, type {}'.format(test_img.shape, test_img.dtype))
	print ('Test image intensity encoded by values from {} to {}'.format(np.min(test_img),np.max(test_img)))

	# initializing arrays to store images
	cars = np.empty((np.concatenate(((len(cars_urls),),test_img.shape))),dtype='uint8')
	noncars = np.empty((np.concatenate(((len(noncars_urls),),test_img.shape))),dtype='uint8')
	print ('Shape of cars array is {}, noncars is {}'.format(cars.shape, noncars.shape))
	
	# loading images in RGB
	for i, fname in enumerate(cars_urls):
		cars[i] = cv2.imread(fname)[:,:,::-1]
	print ('Loaded {} cars'.format(i+1))

	for i, fname in enumerate(noncars_urls):
		noncars[i] = cv2.imread(fname)[:,:,::-1]
	print ('Loaded {} noncars'.format(i+1))
	return cars, noncars


def extract_features(cars, noncars, feats, portion = 1, feature_vec=True):

	print ('Extracting features from each {} images in the dataset'.format(portion))
	test_img = cars[0]
	feat = extract_feature(test_img, feats, feature_vec=feature_vec)
                   
	print ('Feature vector of test image for channel(s) {}, hist_feat {}, spatial_feat {} has shape {}'\
		.format(feats.hog_channel, feats.hist_feat, feats.spatial_feat, feat.shape))

	# filtering images to take portion of them
	indices = [i for i in range(0,len(cars),portion)]
	cars = [cars[i] for i in indices]
	# filtering images to take portion of them
	indices = [i for i in range(0,len(noncars),portion)]
	noncars = [noncars[i] for i in indices]

	#initializing arrays to store features
	cars_feat = np.empty((len(cars), feat.shape[0]), dtype='float64')
	noncars_feat = np.empty((len(noncars), feat.shape[0]), dtype='float64')
	print ('Shape of cars features array is {}, noncars features is {}'.format(cars_feat.shape, noncars_feat.shape))

	for i in range(len(cars)):
	    cars_feat[i] = extract_feature(cars[i], feats, feature_vec=feature_vec)
	print ('Loaded {} cars feature vectors'.format(i+1))
	    
	for i in range(len(noncars)):
	    noncars_feat[i] = extract_feature(noncars[i], feats, feature_vec=feature_vec)
	print ('Loaded {} noncars feature vectors'.format(i+1))
	return cars_feat, noncars_feat
	
def get_dataset(cars_feat, noncars_feat, feats, approach = '80/20'):
	if (approach == '80/20'):
		# preparing training and testing datasets, considering similar time series images
		# taking first 80% as training data, 20% - testing, then shuffling
		cars_80_pers = int(len(cars_feat)*0.8)
		cars_train = cars_feat[:cars_80_pers]
		cars_test = cars_feat[cars_80_pers:]

		noncars_80_pers = int(len(noncars_feat)*0.8)
		noncars_train = noncars_feat[:noncars_80_pers]
		noncars_test = noncars_feat[noncars_80_pers:]

		train = np.concatenate((cars_train, noncars_train))
		test = np.concatenate((cars_test, noncars_test))

		labels_train = np.concatenate((np.ones(len(cars_train)),np.zeros(len(noncars_train))))
		labels_test = np.concatenate((np.ones(len(cars_test)),np.zeros(len(noncars_test))))

		X_train, y_train = shuffle(train, labels_train, random_state=0)
		X_test, y_test = shuffle(test, labels_test, random_state=0)
		
	else:
		# merging cars and noncars feature vectors
		features = np.concatenate((cars_feat, noncars_feat))
		# Fit a per-column scaler
		feats.X_scaler.fit(features)
		# Apply the scaler to X
		scaled_X = feats.X_scaler.transform(features)
		print ('Features vector created with the shape {}'.format(features.shape))

		y = np.concatenate((np.ones((len(cars_feat))), np.zeros((len(noncars_feat)))))
		print ('Labels vector created with the shape {}'.format(y.shape))
		
		X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=42)
	
	print ('Dataset is split into train and test datasets')
	print ('X_train shape {}, X_test shape {}'.format(X_train.shape, X_test.shape))
	return X_train, X_test, y_train, y_test

def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    """
    Function takes an image and returns Histogram of Orientied
    Gradients as a feature vector
    """
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

def bin_spatial(img, size=(32, 32)):
    """
    Function takes image and returns small flatten copy of it
    """
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

def color_hist(img, nbins=32, bins_range=(0, 256)):
    """
    Function takes image, calculates separately intensity histograms
    and returns concatenation of them in form of vector
    """
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

def extract_feature(img, feats, feature_vec=True):
    """
    Function takes image and returns feauture vector
    """
    # Iterate through the list of images
    # apply color conversion if other than 'RGB'
    if feats.cspace != 'RGB':
        if feats.cspace == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif feats.cspace == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif feats.cspace == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif feats.cspace == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif feats.cspace == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)      

    # Call get_hog_features() with vis=False, feature_vec=True
    if feats.hog_channel == 'ALL':
        hog_features = []
        for channel in range(feature_image.shape[2]):
            hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                feats.orient, feats.pix_per_cell, feats.cell_per_block, 
                                vis=False, feature_vec=feature_vec))
        hog_features = np.ravel(hog_features)        
    else:
        hog_features = get_hog_features(feature_image[:,:,feats.hog_channel], feats.orient, 
                    feats.pix_per_cell, feats.cell_per_block, vis=False, feature_vec=feature_vec)
        # Append the new feature vector to the features list
  
    # Return list of feature vector
    return hog_features


def classifier(X_train, X_test, y_train, y_test):
    print ('Training Linear SVM Classifier')
    svc = LinearSVC(C=0.001, random_state=0)
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    return svc

def get_positive_boxes(img, clf, feats):

	# line 1: from bottom to the vanishing point
	a1,b1 = -0.38, 660
	# line 2: from up to the vanishing point
	a2,b2 = 0.09, 360
	# height in [64, 96, 128, 160, 192, 224, 256, 288]
	height = [64+32*i for i in range(8)]

	results = []
	bbox = []
	for h in height:
		# getting the y_min and y_max of region of interest
		x_temp = (h - 300)/-0.47
		y_min = int(a2*x_temp + b2)
		y_max = y_min + h
		scale = 64./h
		# size of the reshaped image
		size = (int(scale*img.shape[1]), 64)

		img_small = img[y_min:y_max, :, :]
		img_small = cv2.resize(img_small, size)

		if feats.hog_channel == 'ALL':
			feats1 = Features()
			feats1.__dict__.update(feats.__dict__)
			feats1.hog_channel=0
			feats2 = Features()
			feats2.__dict__.update(feats.__dict__)
			feats2.hog_channel=1
			feats3 = Features()
			feats3.__dict__.update(feats.__dict__)
			feats3.hog_channel=2

			feat = extract_feature(img_small, feats1, feature_vec=False)
			feat1 = extract_feature(img_small, feats2, feature_vec=False)
			feat2 = extract_feature(img_small, feats3, feature_vec=False)
		else:
			feat = extract_feature(img_small, feats, feature_vec=False)
		            
		# define parameters for sliding window
		shift = 2# amount of cells to step between windows
		bl_p_img = int(size[1]/8) - feats.cell_per_block + 1
		# amount of cells for image
		n_bl = int((feat.shape[1] - bl_p_img)/shift)

		for i in range(n_bl+1):

			if feats.hog_channel == 'ALL':
				x = feat[:,i*shift:bl_p_img+i*shift,:,:,:].ravel()
				x1 = feat1[:,i*shift:bl_p_img+i*shift,:,:,:].ravel()
				x2 = feat2[:,i*shift:bl_p_img+i*shift,:,:,:].ravel()
				x = np.concatenate((x,x1,x2))
			else:
				x = feat[:,i*shift:bl_p_img+i*shift,:,:,:].ravel()
			x = feats.X_scaler.transform(x)
			results.append(int(clf.predict(x)[0]))
			bbox.append(((int(i*shift*feats.pix_per_cell/scale),y_max),(int((size[1]/feats.pix_per_cell+i*shift)*feats.pix_per_cell/scale),y_min)))

	ind_nonzero = np.asarray(results).nonzero()
	bbox_ = np.asarray(bbox)
	pos_boxes = bbox_[ind_nonzero]
	
	#transfering pos_boxes from nd.array to tuple
	pos_boxes_list = [tuple(map(tuple, box)) for box in pos_boxes]
   
	return pos_boxes_list

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    """
    Function takes an image, list of boxes, color and line thickness,
    draws boxes on the image using cv2.rectangle and returns the
    image with the boxes
    """
    img_box = np.copy(img)
    for box in bboxes:
        cv2.rectangle(img_box, box[0], box[1], color, thick)
    return img_box

#def heatmap_img(image, box_list):
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[1][1]:box[0][1], box[0][0]:box[1][0]] += 1
    # Return updated heatmap
    return heatmap# Iterate through list of bboxes

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img
    


class Cars():
	def __init__(self, clf, feats, feature_vec=False):
		self.clf = clf
		self.feats = feats
		self.feature_vec=feature_vec
		self.heat = None
		self.heatmap = None
		self.labels = None
		self.labels_list = []

	def update_heatmap(self,boxes):
		heat = np.zeros_like(self.heat)
		heat = add_heat(heat, boxes)

		ind = np.nonzero(heat)
		self.heat[ind] += 1 

		ind2 = np.ones_like(heat)
		ind2[ind] = 0
		self.heat[np.nonzero(ind2)] -= 1
		self.heat = apply_threshold(self.heat,2)
		self.heat = np.clip(self.heat, 0, 10)

	def draw_labeled_bboxes(self, img, labels):
	# Iterate through all detected cars
		for car_number in np.unique(labels[0]):
			if car_number != 0:
				# Find pixels with each car_number label value
				nonzero = (labels[0] == car_number).nonzero()
				# Identify x and y values of those pixels
				nonzeroy = np.array(nonzero[0])
				nonzerox = np.array(nonzero[1])
				# Define a bounding box based on min/max x and y
				bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
				img = np.array(img)
				# Draw the box on the image
				cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
				# Return the image
		return img

	def labels_thresh(self, labels, heatmap, thresh=3):

		arr = labels[0]
		for car_num in range(1, labels[1]+1):
			heatmap1 = np.where((arr==car_num)==True, heatmap, 0)
			max_heatmap = maximum(heatmap1)
			if max_heatmap < thresh:
				arr = np.where((arr==car_num),0,arr)

		result = (arr, labels[1])
		return result

	def smooth_labels(self, thresh=5):
		label_heat = np.zeros_like(self.heat)
		for label in labels:
			label_heat[np.nonzero(label)] += 1
		label_heat = apply_threshold(label_heat, thresh)
		labels = label(label_heat)
		labels = self.labels_thresh(labels, heatmap, thresh=2)
		return labels

	def pipeline(self, img):
		if self.heat == None:
			self.heat = np.zeros_like(img[:,:,0]).astype(np.float)

		boxes = get_positive_boxes(img, self.clf, self.feats)
		heatmap = add_heat(np.zeros_like(img[:,:,0]).astype(np.float), boxes)
		heatmap = apply_threshold(heatmap, 3)
		self.heatmap = heatmap

		labels = label(heatmap)
		labels = self.labels_thresh(labels, heatmap, thresh=2)
		#self.labels_list.append(labels)
		#if len(self.labels_list)>10:
		#	self.labels_list = self.labels_list[1:]
		#	label_heat = np.zeros_like(img[:,:,0])
		#	for lab in labels:
		#		label_heat[np.nonzero(lab)] += 1
		#	label_heat = apply_threshold(label_heat, 5)
		#	labels = label(label_heat)
		#	labels = self.labels_thresh(labels, label_heat, thresh=2)
			#labels = self.smooth_labels(thresh=5)

		result = self.draw_labeled_bboxes(img, labels)
		return result

class Features():
	def __init__(self, 
			cspace='YUV',
			orient=9,
			pix_per_cell=8,
			cell_per_block=4,
			hog_channel='ALL',
			hist_feat=False,
			spatial_feat=False):
		self.cspace = cspace
		self.orient = orient
		self.pix_per_cell = pix_per_cell
		self.cell_per_block = cell_per_block
		self.hog_channel=hog_channel
		self.hist_feat = hist_feat
		self.spatial_feat = spatial_feat
		self.X_scaler = StandardScaler()

