import cv2
import matplotlib.pyplot as plt
import numpy as np
from class_camera import *
from moviepy.editor import VideoFileClip
import datetime


def ch_threshold(img, ch=0, thresh=(0,180)):
	"""
	Function takes an image (RGB, HSV, etc.),
	channel (0,1,2) for 3 channels, threshold
	and returns binary image of the thresholded channel 
	"""
	if img.shape[2]>ch:
		CH = img[:,:,ch]
		binary = np.zeros_like(CH)
		binary [(CH>thresh[0])&(CH<=thresh[1])] = 1
		return binary
	else:
		return None

def binary_and(binary1, binary2):
    """
    Function takes to binary images same shape and return their intersection
    """
    if binary1.shape == binary2.shape:
        binary = np.where(np.logical_and(binary1,binary2)==True, 1., 0.)
        return binary
    else:
        return None

def binary_or(binary1, binary2):
    """
    Function takes two binary images same shape and return their union
    """
    if binary1.shape == binary2.shape:
        binary = np.where(np.logical_or(binary1,binary2)==True, 1., 0.)
        return binary
    else:
        return None

def binary_substr(img1, img2):
    """
    Function takes two binary images and return binary substaction of second from first
    """
    res = np.copy(img1)
    res[img2==1]=0
    return res

def load_img(fname):
    """
    Function loads file and converts from BGR to RGB
    """
    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def show_2gr(img1,img2,title1='',title2=''):
	"""
	Function takes 2 images and plot them next to each other.
	"""
	f, (ax1,ax2) = plt.subplots(1,2,figsize=(20,10))
	ax1.imshow(img1, cmap='gray')
	ax1.set_title(title1)
	ax2.imshow(img2, cmap='gray')
	ax2.set_title(title2)
	plt.show()

def hist(img):
		"""
		Function takes binary image and returns histogram: 
		sum of "1" pixel for each x on the bottom half of the image
		"""
		img_size = (img.shape[1],img.shape[0])
		histogram = np.sum(img[int(img_size[1]/2):,:], axis=0)
		return histogram

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
	"""
	Function takes 2 images as well as coef-s and adds second 
	image on the first one. Returns result image    
	initial_img * α + img * β + λ
	"""
	return cv2.addWeighted(initial_img, α, img, β, λ)

class Image():
	def __init__(self, image):
		self.img = image
		self.img_size = (self.img.shape[1],self.img.shape[0])

	def abs_sobel_thresh(self,orient='x', thresh_min=0, thresh_max=255):
		"""
		Function takes an image, grayscale it, applies Sobel x or y, 
		then takes an absolute value and applies a threshold.
		"""
		# 1) Convert to grayscale
		gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
		# 2) Take the derivative in x or y given orient = 'x' or 'y'
		if orient!='x':
		    sobel_x = 0
		    sobel_y = 1
		else:
		    sobel_x = 1
		    sobel_y = 0
		sobel = cv2.Sobel(gray, cv2.CV_64F, sobel_x, sobel_y)
		# 3) Take the absolute value of the derivative or gradient
		abs_sobel = np.absolute(sobel)
		# 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
		scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
		# 5) Create a mask of 1's where the scaled gradient magnitude 
		        # is > thresh_min and < thresh_max
		binary_output = np.zeros_like(scaled_sobel)
		binary_output[(scaled_sobel>thresh_min)&(scaled_sobel<=thresh_max)]=1
		# 6) Return this mask as your binary_output image
		return binary_output    

	def mag_thresh(self,sobel_kernel=3, mag_thresh=(0, 255)):
		"""
		Function applies Sobel x and y, then computes the magnitude of the gradient
		and applies a threshold
		"""    
		# 1) Convert to grayscale
		gray = cv2.cvtColor(self.img,cv2.COLOR_RGB2GRAY)
		# 2) Take the gradient in x and y separately
		sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
		sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
		# 3) Calculate the magnitude 
		abs_sobelxy = (sobelx**2+sobely**2)**0.5
		# 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
		scaled_sobelxy = np.uint8(255*abs_sobelxy/np.max(abs_sobelxy))
		# 5) Create a binary mask where mag thresholds are met
		binary_output = np.zeros_like(scaled_sobelxy)
		binary_output[(scaled_sobelxy>mag_thresh[0])&(scaled_sobelxy<mag_thresh[1])] = 1
		# 6) Return this mask as your binary_output image
		return binary_output

	def dir_threshold(self,sobel_kernel=3, thresh=(0, np.pi/2)):
		"""
		Function applies Sobel x and y, then computes the direction of the gradient
		and applies a threshold.
		""" 
		# 1) Convert to grayscale
		gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
		# 2) Take the gradient in x and y separately
		sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
		sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
		# 3) Take the absolute value of the x and y gradients
		abs_sobelx = np.absolute(sobelx)
		abs_sobely = np.absolute(sobely)
		# 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
		direction = np.arctan2(abs_sobely, abs_sobelx)
		# 5) Create a binary mask where direction thresholds are met
		binary_output = np.zeros_like(direction)
		binary_output[(direction>thresh[0])&(direction<thresh[1])] = 1
		# 6) Return this mask as your binary_output image
		return binary_output

	def show_thresh(self,slide=5, step=25, top=200, cols=2, method=0, kernel=3):
		"""
		Function takes an image and applies thresholds in range from 0 to 
		top with sliding window 5 and step 25. The result images are shown in cols columns.
		The following methods are supported:
		method 0: sobel x
		method 1: sobel y
		method 2: sobel magnitude
		method 3: dobel direction
		"""
		num = int((top-step)/slide)+1
		if num%cols == 0:
		    rows = int(num/cols)
		else:
		    rows = int(num/cols)+1

		f,ax = plt.subplots(rows,cols,figsize=(20,int(13*rows/cols)))
		for i in range (num):
		    if method==0:
		        th = self.abs_sobel_thresh(orient='x', thresh_min=i*slide, thresh_max=i*slide+step)
		    elif method==1:
		        th = self.abs_sobel_thresh(orient='y', thresh_min=i*slide, thresh_max=i*slide+step)
		    elif method==2:
		        th = self.mag_thresh(sobel_kernel=kernel, mag_thresh=(i*slide, i*slide+step))
		    elif method==3:
		        th = self.dir_threshold(sobel_kernel=kernel, thresh=(i*slide, i*slide+step)) 
		    ax[int(i/cols),i%cols].imshow(th, cmap='gray')
		    ax[int(i/cols),i%cols].set_title('{}: thresh({}:{})'.format(i,i*slide,i*slide+step))

	def manage_yellow(self):
		"""
		Function applies yellow filter on image and returns the image filtered.
		"""
		# Convert BGR to HSV
		hsv = cv2.cvtColor(self.img, cv2.COLOR_RGB2HSV)

		# define range of yellow color in HSV
		lower_yellow = np.array([20,100,100])
		upper_yellow = np.array([30,255,255])

		# Threshold the HSV image to get only yellow colors
		mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

		return mask

	def manage_grad(self,test=0):
		"""
		Function returns binary thresholded image using gradient feature
		"""
		sx1 = self.abs_sobel_thresh(orient='x', thresh_min=25, thresh_max=55)
		sd_less = self.dir_threshold(sobel_kernel=9, thresh=(1.0721,np.pi/2))
		sd_less2 = self.dir_threshold(sobel_kernel=9, thresh=(0.0982,0.3927))
		sx3 = binary_substr(binary_substr(sx1,sd_less),sd_less2)

		hls = cv2.cvtColor(self.img, cv2.COLOR_RGB2HLS)
		h_less = ch_threshold(hls, 0, (85,256))
		sx3 = binary_substr(sx3, h_less)
		h_less2 = ch_threshold(hls, 0, (-1,6))
		sx3 = binary_substr(sx3, h_less2)
		if test==1:
		    print("nothing to test")
		return sx3    

	def manage_color(self,test=0):
		"""
		Function returns thresholded binary image using color features 
		"""
		hls = cv2.cvtColor(self.img, cv2.COLOR_RGB2HLS)
		s_t150_255 = ch_threshold(hls, 2, (150,255))
		h_t25_180 = ch_threshold(hls, 0, (25,180))
		l_t200_255 = ch_threshold(hls,1, (200,255))
		yellow = self.manage_yellow()
		  
		s_yellow = binary_or(s_t150_255, yellow)
		s_y_h = binary_substr(s_yellow,h_t25_180)
		s_y_h_l = binary_or(s_y_h, l_t200_255)
		return s_y_h_l

	def binary_th(self, test=0):
		if test==1:
		    return binary_or(self.manage_grad(test=1), self.manage_color())
		return binary_or(self.manage_grad(), self.manage_color())

	def hist(self):
		"""
		Function takes binary image and returns histogram: 
		sum of "1" pixel for each x on the bottom half of the image
		"""
		histogram = np.sum(self.img[int(self.img_size[1]/2):,:], axis=0)
		return histogram

	def get_binary_warped(self):
		binary = self.binary_th()

class Lane():
	"""
	Class defines lane and used to detect and track lane on the frames
	"""
	def __init__(self):
		self.leftLine = Line()
		self.rightLine = Line()
		self.cam = Camera()
		self.center = 0
		self.track_image = []

	def pipeline(self,img,debug=0):
		"""
		Pipeline takes the image and brings it through the whole process
		until the lane lines are detected
		"""
		img = self.cam.undist(img)
		#get warped binary image
		binary_warped = self.cam.warp(Image(img).binary_th())
		bw_shape = binary_warped.shape
		
		if (self.leftLine.detected == True and self.rightLine.detected == True):
			self.quick_search(binary_warped,debug)
		else:
			self.blind_search(binary_warped,debug)
	
		if (self.leftLine.fit!=None and self.rightLine.fit!=None):
			polygon = self.fill_lane(bw_shape)
			unwarped_polygon = self.cam.unwarp(polygon)
			# calculate position of lane's center 
			temp = np.nonzero(unwarped_polygon[-1,:,1])[0]
			left, right = temp[0], temp[-1]
			self.center = (int(bw_shape[1]/2) - (int((right-left)/2)+int(left)))*7.4/1280
			img_lines = weighted_img(unwarped_polygon,img, α=1, β=0.5, λ=0.)
			# write text on image
			font = cv2.FONT_HERSHEY_SIMPLEX
			text1 = 'Radius of Curvature: {:.0f}m'.format(np.mean((self.leftLine.radius, self.rightLine.radius)))
			text2 = 'Distance is {:.2f}m {} of center'.format(abs(self.center), 'left' if self.center<0 else 'right')

			cv2.putText(img_lines, text1, (100,100), font, 1,(255,255,255),2)
			cv2.putText(img_lines, text2 ,(100,140), font, 1,(255,255,255),2)
			
			if (debug==1):
				show_2gr(polygon, unwarped_polygon)
				show_2gr(binary_warped, unwarped_polygon)

			return img_lines

		else:
			# no lines detected and not fit available: return original image
			# without lines
			return img

	def blind_search(self, binary_warped, debug):
		"""
		Function used for the initial lines detection using histogram
		and sliding window
		"""
		histogram = hist(binary_warped)
		mid_point = int(len(histogram)/2)
		left = int(np.argmax(histogram[:mid_point]))
		right = int(np.argmax(histogram[mid_point:])) + mid_point

		if debug==1:
			plt.imshow(binary_warped, cmap='gray')
			plt.plot(binary_warped.shape[0]-histogram, linewidth=2)
			print ('histogram left: {}, midpoint: {}, right: {}'.format(left,mid_point,right))
		
		h = binary_warped.shape[0]
		w = binary_warped.shape[1]
		margin = 100
		num_wind = 9
		height = int(binary_warped.shape[0] / num_wind)

		left_line = [[],[]]
		right_line = [[],[]]

		for window in range(num_wind):

		    y_low = h-(window+1)*height
		    y_high = h-window*height

		    left_x_low = left-margin
		    left_x_high = left+margin

		    right_x_low = right-margin
		    right_x_high = right+margin
		    
		    left_box = np.nonzero(binary_warped[y_low:y_high, left_x_low:left_x_high])
		    left_box_y = np.array(left_box[0])+y_low
		    left_box_x = np.array(left_box[1])+left_x_low

		    left_line[0].append(left_box_x)
		    left_line[1].append(left_box_y)

		    right_box = np.nonzero(binary_warped[y_low:y_high, right_x_low:right_x_high])
		    right_box_y = np.array(right_box[0])+y_low
		    right_box_x = np.array(right_box[1])+right_x_low

		    right_line[0].append(right_box_x)
		    right_line[1].append(right_box_y)

		    if len(left_box[1])>50:
		        left = left_x_low+int(np.mean(left_box[1]))
		    if len(right_box[1])>50:
		        right = right_x_low+int(np.mean(right_box[1]))
		
        # rewrite using class attributes
		left_line[0] = np.concatenate(left_line[0])
		left_line[1] = np.concatenate(left_line[1])
		right_line[0] = np.concatenate(right_line[0])
		right_line[1] = np.concatenate(right_line[1])

		self.leftLine.points.append([left_line[0], left_line[1]])  
		self.rightLine.points.append([right_line[0],right_line[1]])

		# polyfit
		# if there are left/right lane lines points, calculate and return
		# polynoms coef-s, else return None
		if (len(left_line[1]>0)&len(right_line[1])>0):
			self.leftLine.detected = True
			self.rightLine.detected = True

			self.leftLine.fit = np.polyfit(left_line[1],left_line[0],2)
			self.rightLine.fit = np.polyfit(right_line[1],right_line[0],2)

			self.leftLine.radius = self.curvature(left_line[0],left_line[1])
			self.rightLine.radius = self.curvature(right_line[0],right_line[1])

			return True
		else:
			return False

	def quick_search(self, binary_warped, debug):
		"""
		Function is used when the lines on the previous frame were detected 
		and searches the line in the sliding window near the line's polynomial
		"""
		nonzero = binary_warped.nonzero()
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		margin = 100
		left_lane_inds = ((nonzerox > (self.leftLine.fit[0]*(nonzeroy**2) + self.leftLine.fit[1]*nonzeroy + self.leftLine.fit[2] - margin)) & (nonzerox < (self.leftLine.fit[0]*(nonzeroy**2) + self.leftLine.fit[1]*nonzeroy + self.leftLine.fit[2] + margin))) 
		right_lane_inds = ((nonzerox > (self.rightLine.fit[0]*(nonzeroy**2) + self.rightLine.fit[1]*nonzeroy + self.rightLine.fit[2] - margin)) & (nonzerox < (self.rightLine.fit[0]*(nonzeroy**2) + self.rightLine.fit[1]*nonzeroy + self.rightLine.fit[2] + margin)))  

		# Again, extract left and right line pixel positions
		leftx = nonzerox[left_lane_inds]
		lefty = nonzeroy[left_lane_inds] 
		rightx = nonzerox[right_lane_inds]
		righty = nonzeroy[right_lane_inds]
		# Fit a second order polynomial to each
		if len(leftx)>0 and len(rightx)>0:
			self.leftLine.fit = np.polyfit(lefty, leftx, 2)
			self.rightLine.fit = np.polyfit(righty, rightx, 2)
			self.leftLine.radius = self.curvature(leftx,lefty)
			self.rightLine.radius = self.curvature(rightx,righty)
			return True
		else:
			self.leftLine.detected = False
			self.rightLine.detected = False
			return False
		

	def curvature(self,x, y):
		# Define conversions in x and y from pixels space to meters
		ym_per_pix = 50/720 # meters per pixel in y dimension
		xm_per_pix = 7.4/1280 # meters per pixel in x dimension

		y_eval = np.max(y)

		# Fit new polynomials to x,y in world space
		fit_cr = np.polyfit(y*ym_per_pix, x*xm_per_pix, 2)

		# Calculate the new radii of curvature
		curverad = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
		return curverad

	def fill_lane(self,img_shape):
		"""
		Function takes 2 lines' polynomials and return the image of the filled with green polygon 
		between 2 lines
		"""

		binary_l = np.zeros(img_shape, dtype=np.uint8)

		ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
		plotx_l = self.leftLine.fit[0]*ploty**2 + self.leftLine.fit[1]*ploty + self.leftLine.fit[2]
		plotx_r = self.rightLine.fit[0]*ploty**2 + self.rightLine.fit[1]*ploty + self.rightLine.fit[2]

		line_points_l = np.column_stack((plotx_l,ploty))
		line_points_r = np.column_stack((plotx_r,ploty))
		line_points = np.concatenate((line_points_l,line_points_r[::-1],line_points_l[:1]))

		cv2.fillPoly(binary_l, np.int32([line_points]),color=255)

		polygon = np.dstack((np.zeros(img_shape),binary_l,np.zeros(img_shape))).astype('uint8')
		
		return polygon
		unwarped_polygon = self.cam.unwarp(polygon)
		return unwarped_polygon

class Line():
	"""
	Class defines lane line
	"""
	def __init__(self):
		self.detected = False
		self.fit = None
		self.radius = None
		self.points = []
	
	def get_line(self):
		if self.detected == False:
			self.blind_search()
		else: 
			self.quick_search()
		return self.detected

	def fit_line(self,points):
		if len(points[0]>0):
			self.polynomial = np.polyfit(self.points[1],self.points[0],2)
			self.detected = True 

def process_video(lane, fname, output):
	"""
	function takes object lane, url of the video and name of 
	the output video and process the given video in order to detect
	and visualize lane. 
	"""
	clip = VideoFileClip(fname)
	output_name = output
	output_clip = clip.fl_image(lane.pipeline)
	output_clip.write_videofile(output_name, audio=False)
	print ('Video processed successfully')
