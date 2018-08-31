# Calculation functions
# Functions that perform numerical analysis on lane lines to find lane curvatures, distance from either lane line, and approximate velocity

import numpy as np
import cv2

# Calculates the radii of lane curvatures on a warped image by taking derviatives of a second-order polynomial and converting to real-world coordinates
def curvature_analysis(raw_image,left_px,right_px):
    x_m_per_pix=3.7/750 # m per pixel in x-dimension
    y_m_per_pix=27/720 # m per pixel in y-dimension
    
    left_px=np.array(left_px)[1,:].T
    right_px=np.array(right_px)[1,:].T
    
    left_len=len(left_px)
    right_len=len(right_px)
    
    y_grid_left=np.linspace(0,raw_image.shape[0]-1,left_len)
    y_grid_right=np.linspace(0,raw_image.shape[0]-1,right_len)
    
    y_eval_one=np.max(y_grid_left)*y_m_per_pix # In real-world coordinates
    y_eval_two=y_eval_one/2
    
    # Corrected polynomial fit
    leftfit_corrected=np.polyfit(y_grid_left*y_m_per_pix,left_px*x_m_per_pix,2)
    rightfit_corrected=np.polyfit(y_grid_right*y_m_per_pix,right_px*x_m_per_pix,2)
    
    # Left Curvature Calculation
    left_curve_one=((1+(2*leftfit_corrected[0]*y_eval_one+leftfit_corrected[1])**2)**1.5)/np.absolute(2*leftfit_corrected[0])
    #left_curve_two=((1+(2*leftfit_corrected[0]*y_eval_two+leftfit_corrected[1])**2)**1.5)/np.absolute(2*leftfit_corrected[0])
    
    # Right Curvature Calculation
    right_curve_one=((1+(2*rightfit_corrected[0]*y_eval_one+rightfit_corrected[1])**2)**1.5)/np.absolute(2*rightfit_corrected[0])
    #right_curve_two=((1+(2*rightfit_corrected[0]*y_eval_two+rightfit_corrected[1])**2)**1.5)/np.absolute(2*rightfit_corrected[0])
    
    # Averaging curvature calculations
    avg_left_curve=1.*(left_curve_one)
    avg_right_curve=1.*(right_curve_one)
    #avg_left_curve=0.5*(left_curve_one+left_curve_two)
    #avg_right_curve=0.5*(right_curve_one+left_curve_two)
    
    avg_curve=0.5*(avg_left_curve+avg_right_curve)
    
    if ((left_curve_one is not None) or (right_curve_one is not None)):
        print("Left Lanes {} m".format(round(left_curve_one,4)))
        print("Right Lanes {} m".format(round(right_curve_one,4)))
    else:
        print("Left Lanes {} m".format(left_curve_one))
        print("Right Lanes {} m".format(right_curve_one))
    #if ((left_curve_one is not None) or (left_curve_two is not None) or (right_curve_one is not None) or (right_curve_two is not None)):
    #    print("Left Lanes {} m, {} m".format(round(left_curve_one,4),round(left_curve_two,4)))
    #    print("Right Lanes {} m, {} m".format(round(right_curve_one,4),round(right_curve_two,4)))
    #else:
    #    print("Left Lanes {} m, {} m".format(left_curve_one,left_curve_two,4))
    #    print("Right Lanes {} m, {} m".format(right_curve_one,right_curve_two,4))
    
    return avg_left_curve,avg_right_curve,avg_curve

# Figures out offset of of the car camera from the center of the image/warped image
def offset_analysis(raw_image, left_px, right_px, left_fit, right_fit):
    h,w=raw_image.shape
    
    x_m_per_pix=3.7/750 # m per pixel in x-dimension
    car_center=0.5*raw_image.shape[1]
    
    left_lane_pos=(left_fit[0]*h**2)+(left_fit[1]*h)+left_fit[2]
    right_lane_pos=(right_fit[0]*h**2)+(right_fit[1]*h)+right_fit[2]
    midpoint_lanes=0.5*(left_lane_pos+right_lane_pos)
    
    if (car_center>=midpoint_lanes):
        bias='R'
        offset=(car_center-midpoint_lanes)*x_m_per_pix
    else:
        bias='L'
        offset=(midpoint_lanes-car_center)*x_m_per_pix
        
    return bias,offset

# Obtains a velocity estimate based on changing detection of either right or left lanes for a video pipeline
def velocity_analysis(fps):
    c=-1
    return c    