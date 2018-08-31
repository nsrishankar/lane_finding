# Video pipeline class that is self-contained (just needs the other function python scripts)

import math
import os
import pickle
import numpy as np
import cv2
import glob
from collections import deque
from camera_image_calibration import undistort, birds_eye_transform, remap, annotate
from image_thresholding import edge_detection
from lane_detection import visualize_windowfit, combined_windowfit_filled
from analysis import curvature_analysis, offset_analysis

# Load saved pickled data for matrix and distortion coefficients
pickle_path='undistortion_pickle.p'

with open(pickle_path,mode='rb') as f:
        undistort_data=pickle.load(f)
undistort_camera_matrix,undistort_coefficients=undistort_data['dist_matrix'], undistort_data['dist_coefficients']

# Class to obtain characteristics of line detection, and to create a pipeline
class Lane_finding:
    def __init__(self,n_iterations=15):
        
        self.detected_left=False # Was left-line detected in the last iteration
        self.detected_right=False # Was right-line detected in the last iteration
        
        self.recent_xfitted_left=deque(maxlen=n_iterations) # X-values of fitted left-line over last n-iterations
        self.recent_xfitted_right=deque(maxlen=n_iterations) # X-values of fitted right-line over last n-iterations
        
        self.best_fit_left=None # Mean Left-line polyfit coefficients over last n-iterations
        self.best_fit_right=None # Mean Right-line polyfit coefficients over last n-iterations
        
        self.current_fit_left=None # Current fit of Left-line polynomial
        self.current_fit_right=None # Current fit of Right-line polynomial
        
        self.current_valid_curve=None # Was curve-information detected
        self.curvature_left=None # Radius of curvature of left line
        self.curvature_right=None # Radius of curvature of right line
        self.curvature=None # Averaged curvature of left, right lines
        
        self.offset=None # Offset of vehicle center from lane center
        self.bias=None # Vehicle Bias

        self.diff_coeffs=[] # Difference in fit coefficients between last and new fit
        self.diff_curve=[] # Difference in curve values between last and new fit
        
        self.left_px_all=deque(maxlen=n_iterations) # Detected line pixels over last n-iterations
        self.right_px_all=deque(maxlen=n_iterations) # Detected line pixels over last n-iterations
        
    # Sanity Checks
    #Checking that they have similar curvature
    #Checking that they are separated by approximately the right distance horizontally
    #Checking that they are roughly parallel
    
    def sanity_fit(self,prev_coeffs,current_coeffs):
        difference=np.absolute((prev_coeffs-current_coeffs)/(prev_coeffs))
        print("Sanity Fit Difference", difference)
        self.diff_coeffs.append(difference)
        
        if not ([(difference[0]<1.) and (difference[1]<1.) and (difference[2]<1.)]):
            print("Polyfitting coefficients changed by:", difference)
        return ([(difference[0]<1.) and (difference[1]<1.) and (difference[2]<1.)]) 

    def sanity_curve(self,prev_curve,current_curve):
        difference=np.absolute((prev_curve-current_curve)/(prev_curve))
        print("Sanity Curve Difference", difference)
        self.diff_curve.append(difference)
        
        if not ([(difference<5.)]):
            print("Curvature coefficients changed by:", difference)
        return ([(difference<5.)])
                   
    def video_pipeline(self,raw_image,undistort_matrix=undistort_camera_matrix,undistort_coefficients=undistort_coefficients):
        
        # Make a copy of initial raw image
        #copy=np.copy(raw_image)

        # Correction for camera distortion and resizing raw_image
        undistorted_image=undistort(cv2.resize(raw_image,(1280,720)),undistort_matrix,undistort_coefficients)
        
        # HLS S-Channel color and Sobel-edge thresholding
        color_binary, combined_binary=edge_detection(undistorted_image)
        
        # Perform a perspective transformation on binary image to obtain a birds-eye perspective
        top_combined_binary, M, Minv=birds_eye_transform(combined_binary)
        
        try:
            # Fit polynomials to detected left and right lanes using a sliding window fit (and previous window search)
            # If a fit is detected.
            left_fit_combined, right_fit_combined, left_px, right_px=combined_windowfit_filled(top_combined_binary)
        except TypeError as error:
            print("Type Error. Window fit doesn't return a lane line polynomial fit.")
            left_fit_combined=None
            right_fit_combined=None
            left_px=None
            right_px=None
        print("Left fit combined:", left_fit_combined)
        print("Left_px:", left_px)

        # Curvature and Bias analysis
        if ((left_fit_combined is not None) and (right_fit_combined is not None)):
            try:
                avg_left_curve, avg_right_curve, avg_curve=curvature_analysis(top_combined_binary,left_px,right_px)
                bias, offset=offset_analysis(top_combined_binary,left_px,right_px,left_fit_combined,right_fit_combined)
                
                #print("Avg_left_curve:", avg_left_curve)
                #print("Avg_right_curve:", avg_right_curve)
                #print("Avg_curve:", avg_curve)
                #print("Bias:", bias)
                #print("Offset:", offset)
                
                if bias=='R':
                    disp_bias="--> (right lane)"
                else:
                    disp_bias="<-- (left lane)"

                if ((self.current_valid_curve is None)):
                    print("Curve- 01 loop")
                    self.current_valid_curve=True
                    self.curvature_left=avg_left_curve
                    self.curvature_right=avg_right_curve
                    self.curvature=avg_curve

                    self.offset=offset
                    self.bias=disp_bias

                elif (self.sanity_curve(self.curvature,avg_curve)): 
                    print("Curve- 02 loop")
                    self.current_valid_curve=True
                    self.curvature_left=avg_left_curve
                    self.curvature_right=avg_right_curve
                    self.curvature=avg_curve

                    self.offset=offset
                    self.bias=disp_bias


                else:
                    # Sanity check failed
                    print("Curve- sanity failed")
                    self.current_valid_curve=False

            except:
                # Using previous data
                print("Except: Using previous data")
                self.current_valid_curve=False
                avg_left_curve=self.curvature_left
                avg_right_curve=self.curvature_right
                avg_curve=self.curvature

                offset=self.offset
                disp_bias=self.bias

        else:
            # Using previous data
            print("No polyfit: Using previous data")
            self.current_valid_curve=False
            avg_left_curve=self.curvature_left
            avg_right_curve=self.curvature_right
            avg_curve=self.curvature

            offset=self.offset
            disp_bias=self.bias

        # Storing valid left-line polynomial fit data
        if ((left_fit_combined is not None) and (len(left_px)!=0)):
            if self.current_fit_left is None:
                print("Left- 01 loop")
                self.detected_left=True
                self.current_fit_left=left_fit_combined
                self.best_fit_left=left_fit_combined
                
                self.recent_xfitted_left.append(left_fit_combined)
                self.left_px_all.append(left_px)

            elif ((self.sanity_fit(self.current_fit_left,left_fit_combined)) and (self.current_valid_curve)):
                print("Left- 02 loop")
                self.detected_left=True
                self.current_fit_left=left_fit_combined
                self.best_fit_left=np.mean(self.recent_xfitted_left,axis=0)
                
                self.recent_xfitted_left.append(left_fit_combined)
                self.left_px_all.append(left_px)
            
            else:
                # Sanity check failed
                print("Left- sanity failed")
                self.detected_left=False

        elif left_fit_combined is None:
            # No polynomial coefficients found
            print("Left- no polyfit")
            self.detected_left=False
            
        # Storing valid right-line polynomial fit data
        if ((right_fit_combined is not None) and (len(right_px)!=0)):
            if self.current_fit_right is None:
                print("Right- 01 loop")
                self.detected_right=True
                self.current_fit_right=right_fit_combined
                self.best_fit_right=right_fit_combined
                
                self.recent_xfitted_right.append(right_fit_combined)
                self.right_px_all.append(right_px)

            elif ((self.sanity_fit(self.current_fit_right,right_fit_combined)) and (self.current_valid_curve)):
                print("Right- 02 loop")
                self.detected_right=True
                self.current_fit_right=right_fit_combined
                self.best_fit_right=np.mean(self.recent_xfitted_right,axis=0)
                
                self.recent_xfitted_right.append(right_fit_combined)
                self.right_px_all.append(right_px)
            
            else:
                # Sanity check failed
                print("Right- sanity failed")
                self.detected_right=False

        elif right_fit_combined is None:
            # No polynomial coefficients found
            print("Right- no polyfit")
            self.detected_right=False
            
        #print("Self.detected_left:",self.detected_left)
        #print("Self.detected_right:",self.detected_right)
        #print("Self.best_fit_left:",self.best_fit_left)
        #print("Self.best_fit_right:",self.best_fit_right)
        #print("Self.curvature_left:",self.curvature_left)
        #print("Self.curvature_right:",self.curvature_right)
        
        if ((self.best_fit_left is not None and self.best_fit_right is not None) or (self.detected_left and self.detected_right)):
            
            # Overlay lane edge polynomials and lane polygons
            combined_image=remap(undistorted_image,top_combined_binary,Minv,self.best_fit_left,self.best_fit_right)

            # Annotate overlayed image with lane information
            annotated_image=annotate(combined_image,self.curvature_left,self.curvature_right,
                                      self.curvature,self.bias,self.offset)
            return annotated_image
        else:
            print("Lane line not detected or no best fit line retrieved.")
            return undistorted_image