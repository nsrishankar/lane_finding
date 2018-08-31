# Color Space Transformation functions and gradient detection
# Functions that takes in a set of chessboard images to output a camera matrix and distortion coefficients 

import numpy as np
import cv2

# Converts from BGR- space to HSL space 
def hls_conversion(raw_image):
    return cv2.cvtColor(raw_image,cv2.COLOR_BGR2HLS)

# Converts from BGR- space to HSV space 
def hsv_conversion(raw_image):
    return cv2.cvtColor(raw_image,cv2.COLOR_BGR2HSV)

# Converts from BGR- space to grayscale
def grayscale_conversion(raw_image):
    return cv2.cvtColor(raw_image,cv2.COLOR_BGR2GRAY)

# Stacks a HLS- space image from modified S-channel to existing HL-channels, and then converts to grayscale
#def hls_stack2_gray(s_channel,hl_channel):
#    hls=np.dstack((hl_channel,s_channel))
#    bgr=cv2.cvtColor(hls,cv2.COLOR_HLS2BGR)
#    gray=grayscale(bgr)
#    return gray

# Sets a threshold to Saturation-channel images
#def color_threshold(raw_image,threshold=(170,255)):
#    s_channel_binary=np.zeros_like(raw_image)
#    s_channel_binary[(raw_image>=threshold[0]) & (raw_image<=threshold[1])]=1
#    return s_channel_binary

def color_threshold(raw_image, sthreshold=(170,255),lthreshold=(100,255)):
    denoised_image=cv2.fastNlMeansDenoisingColored(raw_image,None,10,10,7,21)
    hls=hls_conversion(denoised_image)
    hsv=hsv_conversion(denoised_image)
    h_channel=hls[:,:,0]
    l_channel=hls[:,:,1]
    s_channel=hls[:,:,2]
    
    white_low_threshold=np.uint8([10,200,0])
    white_high_threshold=np.uint8([255,255,255])
    yellow_low_threshold=np.uint8([15,60,130])
    yellow_high_threshold=np.uint8([150,255,255])
    
    white_lanes=cv2.inRange(hls,white_low_threshold,white_high_threshold)
    yellow_lanes=cv2.inRange(hsv,yellow_low_threshold,yellow_high_threshold)
    combined_lanes=cv2.bitwise_or(white_lanes,yellow_lanes)
    
    s_channel_binary=np.zeros_like(s_channel)
    s_channel_binary[(s_channel>=sthreshold[0]) & (s_channel<=sthreshold[1])]=1
    
    l_channel_binary=np.zeros_like(l_channel)
    l_channel_binary[(l_channel>=lthreshold[0]) & (l_channel<=lthreshold[1])]=1
    
    combined=np.zeros_like(s_channel)
    combined[((s_channel_binary==1) & (l_channel_binary==1)) | (combined_lanes>0)]=1

    return combined
# Returns Sobel edges for either x- or y-directions within a certain threshold
def sobel_edge(raw_image,sobel_kernel=3,orient='x',threshold=(170,255)):
    if orient=='x':
        sobel=cv2.Sobel(raw_image,cv2.CV_64F,1,0,ksize=sobel_kernel)
    else:
        sobel=cv2.Sobel(raw_image,cv2.CV_64F,0,1,ksize=sobel_kernel)
    
    abs_sobel=np.absolute(sobel)
    scaled=((255*abs_sobel)/np.max(abs_sobel)).astype(np.uint8)
    
    mask=np.zeros_like(scaled)
    mask[(scaled>=threshold[0]) & (scaled<=threshold[1])]=1
    return mask

# Returns magnitude of Sobel edges within a certain magnitude threshold
def sobel_magnitude(raw_image,sobel_kernel=3,threshold=(0,255)):
    sobel_x=cv2.Sobel(raw_image,cv2.CV_64F,1,0,ksize=sobel_kernel)
    sobel_y=cv2.Sobel(raw_image,cv2.CV_64F,0,1,ksize=sobel_kernel)
    
    magnitude_sobel=np.sqrt(np.power(sobel_x,2)+np.power(sobel_y,2))
    scaled=((255*magnitude_sobel)/np.max(magnitude_sobel)).astype(np.uint8)
    
    mask=np.zeros_like(scaled)
    mask[(scaled>=threshold[0]) & (scaled<=threshold[1])]=1
    return mask

# Returns the direction of Sobel edges within a certain angle threshold
def sobel_direction(raw_image,sobel_kernel=3,threshold=(0,0.5*np.pi)):
    sobel_x=cv2.Sobel(raw_image,cv2.CV_64F,1,0,ksize=sobel_kernel)
    sobel_y=cv2.Sobel(raw_image,cv2.CV_64F,0,1,ksize=sobel_kernel)
    
    direction_sobel=np.absolute(np.arctan2(sobel_y,sobel_x))
    
    mask=np.zeros_like(direction_sobel)
    mask[(direction_sobel>=threshold[0]) & (direction_sobel<=threshold[1])]=1
    return mask
                            
def edge_detection(raw_image):
    image_s_channel=hls_conversion(raw_image)[:,:,2]
    image_hl_channels=hls_conversion(raw_image)[:,:,0:2]
    
    sobel_x=sobel_edge(image_s_channel,sobel_kernel=5,orient='x',threshold=(20,100))
    sobel_y=sobel_edge(image_s_channel,sobel_kernel=5,orient='y',threshold=(20,100))
    sobel_xy_magnitude=sobel_magnitude(image_s_channel,sobel_kernel=3,threshold=(20,100))
    sobel_xy_angle=sobel_direction(image_s_channel,sobel_kernel=3,threshold=(0.85,1.05))
                            
    sobel_mask=np.zeros_like(image_s_channel)
    sobel_mask[((sobel_x==1) & (sobel_y==1)) | ((sobel_xy_magnitude==1) & (sobel_xy_angle==1))]=1

    s_channel_binary=color_threshold(raw_image,sthreshold=(170,255),lthreshold=(100,255))
    
    sobel_mask_scaled=(255*sobel_mask/np.max(sobel_mask)).astype(np.uint8)
    s_channel_binary_scaled=(255*s_channel_binary/np.max(s_channel_binary)).astype(np.uint8)
    color_binary=np.dstack((np.zeros_like(image_s_channel),sobel_mask_scaled,s_channel_binary_scaled))
    
    combined_binary=np.zeros_like(sobel_mask)
    combined_binary[(sobel_mask==1) | (s_channel_binary==1)]=1
    
    return color_binary, combined_binary