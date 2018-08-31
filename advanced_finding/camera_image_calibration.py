# Image Calibrations and Viewpoint transformations
# Functions that takes in a set of chessboard images to output a camera matrix and distortion coefficients 

import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt

# Returns the corners of an image
def calibrate_camera(images_path, pattern=(9,6), draw_chessboard=True):
    print("Camera Calibration initialized.")
    images_path=glob.glob(images_path)
    obj_points=np.zeros((pattern[1]*pattern[0],3),np.float32)
    obj_points[:,:2]=np.mgrid[0:pattern[0],0:pattern[1]].T.reshape(-1,2)
    
    # Array objects
    obj_array=[] # 3D points in real-world object space
    img_array=[] # 2D points in image space
    
    # Plotting
    fig, ax=plt.subplots(5,4, figsize=(10, 10))
    ax=ax.ravel()
    cal_images=[]
    uncal_images=[]
    for i, path in enumerate(images_path):
        image=cv2.imread(path)
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        
        # Chessboard Corners
        ret, corners=cv2.findChessboardCorners(gray,(pattern[0],pattern[1]),None)
        
        if ret:
            obj_array.append(obj_points)
            img_array.append(corners)
            print("Detected corners for image {}.".format(i))
            cal_images.append(image)
            
        else:
            uncal_images.append(image)
            
        if draw_chessboard:
            # print("Corrected images with detected corners.")
            for image in cal_images:
                image=cv2.drawChessboardCorners(image,pattern,corners,ret)
                ax[i].axis('off')
                ax[i].imshow(image)
    print("(9,6) pattern corners cannot be detected in {} images in the camera_cal folder.".format(len(np.asarray(uncal_images))))
    return obj_array,img_array,cal_images,uncal_images

# Takes in a raw_image, object and image corner/array to undistort the image.
def undistortion(raw_image, object_points, image_points):
    image_shape=(raw_image.shape[0],raw_image.shape[1])
    
    ret,camera_matrix,distortion_coefficients,rvecs,tvecs=cv2.calibrateCamera(object_points, image_points, image_shape,
                                                                              None, None)
    destination_image=cv2.undistort(raw_image, camera_matrix, distortion_coefficients, None, camera_matrix)
    
    return destination_image, camera_matrix, distortion_coefficients

# Define undistortion for any future images and videos after chessboard calibration
def undistort(raw_image, camera_matrix,distortion_coefficients):
    return cv2.undistort(raw_image, camera_matrix, distortion_coefficients, None, camera_matrix)

# Perspective Transformation to view images from the top-down.
def birds_eye_transform(raw_image):
    y_len,x_len=raw_image.shape[0:2]

    #src_points=np.float32([[-100,y_len],[x_len+100,y_len],
    #                      [0.5*x_len-100,0.6*y_len],[0.5*x_len+100,0.6*y_len]])
    src_points=np.float32([[0,y_len],[x_len-0,y_len],
                          [0.55*x_len-110,0.6*y_len],[0.55*x_len+20,0.6*y_len]])

    dest_points=np.float32([[100,y_len],[x_len-100,y_len],
                            [100,0],[x_len-100,0]])
        
    warp_matrix=cv2.getPerspectiveTransform(src_points,dest_points)
    inverse_warp_matrix=cv2.getPerspectiveTransform(dest_points,src_points)
    
    warped_image=cv2.warpPerspective(raw_image,warp_matrix,(x_len,y_len),flags=cv2.INTER_LINEAR)
    return warped_image, warp_matrix, inverse_warp_matrix


def annotate(raw_image,avg_left_curve,avg_right_curve,avg_curve,bias,offset):
    image=np.copy(raw_image)
    
    if bias=='R':
        disp_bias="--> (right lane)."
    else:
        disp_bias="<-- (left lane)."
    
    font=cv2.FONT_HERSHEY_SIMPLEX
    if ((avg_left_curve is not None) or (avg_right_curve is not None) or (avg_curve is not None) or (offset is not None)): 
        curve_details="Left Curve: {} m. Right Curve: {} m.".format(round(avg_left_curve,3),round(avg_right_curve,3))
        average_curve_details="Average Curve: {} m.".format(round(avg_curve,3))
        bias_details="Offset: {} m. Towards {}".format(round(offset,3),disp_bias)
    else:
        curve_details="Left Curve: {} m. Right Curve: {} m.".format(avg_left_curve,avg_right_curve)
        average_curve_details="Average Curve: {} m.".format(avg_curve)
        bias_details="Offset: {} m. Towards {}".format(offset,disp_bias)

    cv2.putText(image,curve_details, (40,50), font, 1.5, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(image,average_curve_details, (40,100), font, 1.5, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(image,bias_details, (40,150), font, 1.5, (255,255,255), 2, cv2.LINE_AA)
    
    return image

# Take a raw image from the birds-eye/top-down perspective and warp back into the original projection using inverse warp from destination to source points.
def remap(original_image, transformed_image,inverse_warp_matrix,left_fit,right_fit):
    y_len,x_len=transformed_image.shape[0:2]
    copy=np.copy(original_image)
    
    blank_1ch=np.zeros_like(transformed_image).astype(np.uint8)
    blank=np.dstack((blank_1ch,blank_1ch,blank_1ch))
    
    y_grid=np.linspace(int(0.55*y_len),y_len-1,2*y_len)
    left_lane=(left_fit[0]*y_grid**2)+(left_fit[1]*y_grid)+left_fit[2]
    right_lane=(right_fit[0]*y_grid**2)+(right_fit[1]*y_grid)+right_fit[2]
    
    left_points=np.vstack([left_lane,y_grid]).T
    right_points=np.vstack([right_lane,y_grid]).T
    
    points=np.int32([np.vstack([left_points,np.flipud(right_points)])])
    
    cv2.fillConvexPoly(blank,points,(0,255,0))
    cv2.polylines(blank,np.int_([left_points]),False,(255,0,255),20)
    cv2.polylines(blank,np.int_([right_points]),False,(255,0,255),20)
    
    unwarped_image=cv2.warpPerspective(blank,inverse_warp_matrix,(x_len,y_len),flags=cv2.INTER_LINEAR)
    
    combine=cv2.addWeighted(unwarped_image,0.5,copy,1.0,0)
    
    return combine