import numpy as np
import cv2
import glob

class Camera():
    def __init__(self):
        self.mtx, self.dist = self.camera_calibration()
        self.M_warp, self.M_unwarp = self.get_warp_matrix()

    def camera_calibration(self):
        """
        Function uses chess board images to calculate 
        calibration coefficients and returns mtx, dist
        """
        objpoints = []
        imgpoints = []
        objp = np.zeros((6*9,3),np.float32)
        objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

        url = '../camera_cal/'
        name = 'calibration*.jpg'

        images = glob.glob(url+name)

        if len(images)>0:
            for image in images:
                img = cv2.imread(image)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
                if ret == True:
                    imgpoints.append(corners)
                    objpoints.append(objp)

            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
            return mtx, dist
        else:
            return None, None

    def undist (self,img):
        """
        Function takes an image and mske distortion correction
        """
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

    def get_warp_matrix(self):
        w,h = 1280,720
        x,y = 0.5*w, 0.8*h

        src = np.float32([[200./1280*w,720./720*h],
                      [453./1280*w,547./720*h],
                      [835./1280*w,547./720*h],
                      [1100./1280*w,720./720*h]])
        dst = np.float32([[(w-x)/2.,h],
                      [(w-x)/2.,0.82*h],
                      [(w+x)/2.,0.82*h],
                      [(w+x)/2.,h]])

        M_warp = cv2.getPerspectiveTransform(src,dst)
        M_unwarp = cv2.getPerspectiveTransform(dst,src)
        return M_warp, M_unwarp

    def warp(self, img):
        """
        Function takes an image and return an image with perspective transformation applied.
        Src points were taken on the images, provided by Udacity. Depends from the camera.
        Proportion is not saved: height ~22.5m, width ~7.4m
        """
        img_size = (img.shape[1], img.shape[0])
        img_warp = cv2.warpPerspective(img,self.M_warp,img_size,flags=cv2.INTER_LINEAR)
        return img_warp

    def unwarp(self, img):
        """
        Function takes warped image and returns unwarped image
        """
        img_size = (img.shape[1], img.shape[0])
        img_unwarp = cv2.warpPerspective(img,self.M_unwarp,img_size,flags=cv2.INTER_LINEAR)
        return img_unwarp