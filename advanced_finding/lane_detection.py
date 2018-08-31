# Lane detection and enhancement functions
# Functions that take a warped image with detected edges to properly superimpose lane information

import numpy as np
import cv2
import matplotlib.pyplot as plt

# Shows a lateral cross-section of an image displaying areas that activate the thresholding
def pixel_histogram(raw_image):
    y_len=raw_image.shape[0]
    histogram=np.sum(raw_image[y_len//4:,:],axis=0)
    return histogram

def new_sliding_window(raw_binary_warped_image,nwindows=12,window_margin=80,min_pixel=50):
    # nwindows: Number of sliding windows
    # windows_margin: Width of a window's +/- margin
    # min_pixel: Number of minimum pixels needed to recenter a window
    y_len=raw_binary_warped_image.shape[0]
    
    histogram=pixel_histogram(raw_binary_warped_image)
    #print("Hist shape", histogram.shape)
    
    # Since there might be noise in the corners of the image, its good to find maximum between pixel points (0.25,0.5) units and (0.5,0.75) units.
    midpoint=np.int(histogram.shape[0]/2)
    #print("Midpoint", midpoint)
    quarterpoint=np.int(midpoint/2)
    left_max = np.argmax(histogram[(quarterpoint-100):midpoint])+(quarterpoint-100)
    #print("Left max", left_max)
    right_max = np.argmax(histogram[midpoint:(midpoint+quarterpoint+100)])+(midpoint)
    #print("Right max", right_max)

    # Sliding windows details/height
    window_data=[]
    window_height=np.int(y_len/nwindows)
    # Non-zero pixels in the binary_image
    nonzero_pixels=raw_binary_warped_image.nonzero()
    nonzero_pixels_x=np.array(nonzero_pixels[1])
    nonzero_pixels_y=np.array(nonzero_pixels[0])
    
    # Current positions
    left_x_current=left_max
    right_x_current=right_max
    
    # Obtain left and right lane pixel indices
    left_lane_indx=[]
    right_lane_indx=[]
    
    # Windows
    for window in range(nwindows):
        window_y_low=y_len-(window+1)*window_height
        window_y_high=y_len-window*window_height
        
        window_x_left_low=left_x_current-window_margin
        window_x_left_high=left_x_current+window_margin
        
        window_x_right_low=right_x_current-window_margin
        window_x_right_high=right_x_current+window_margin
        
        window_data.append((window_y_low,window_y_high,
                            window_x_left_low,window_x_left_high,
                            window_x_right_low,window_x_right_high))
        # Draw Rectangle
        #cv2.rectangle(output_image,(window_x_left_low,window_y_low),(window_x_left_high,window_y_high),(0,255,0),2)
        #cv2.rectangle(output_image,(window_x_right_low,window_y_low),(window_x_right_high,window_y_high),(0,255,0),2)
        
        # Nonzero pixels within a window
        window_left_activated=((nonzero_pixels_y>=window_y_low) & (nonzero_pixels_y<window_y_high) &
                                (nonzero_pixels_x>=window_x_left_low) & (nonzero_pixels_x<window_x_left_high))
        window_left_indx=window_left_activated.nonzero()[0]
        
        window_right_activated=((nonzero_pixels_y>=window_y_low) & (nonzero_pixels_y<window_y_high) &
                                (nonzero_pixels_x>=window_x_right_low) & (nonzero_pixels_x<window_x_right_high))
        window_right_indx=window_right_activated.nonzero()[0]
        
        # Append data
        left_lane_indx.append(window_left_indx)
        right_lane_indx.append(window_right_indx)
        
        if len(window_left_indx)>=min_pixel: # Min_pixel --> Number of valid candidates needed to recenter a window
            left_x_current=np.int(np.mean(nonzero_pixels_x[window_left_indx]))
        if len(window_right_indx)>=min_pixel:
            right_x_current=np.int(np.mean(nonzero_pixels_x[window_right_indx]))
        
    # Concatenate data
    left_lane_indx=np.concatenate(left_lane_indx)
    right_lane_indx=np.concatenate(right_lane_indx)
    
    # Extract pixel data
    left_x=nonzero_pixels_x[left_lane_indx]
    left_y=nonzero_pixels_y[left_lane_indx]
    right_x=nonzero_pixels_x[right_lane_indx]
    right_y=nonzero_pixels_y[right_lane_indx]
    
    # Combining pixel data
    left_px=[left_y,left_x]
    right_px=[right_y,right_x]
    
    # Polyfit a second-order polynomial to each pixel information
    left_fit=None
    right_fit=None
    if len(left_x)!=0:
        left_fit=np.polyfit(left_y,left_x,2)
    if len(right_x)!=0:
        right_fit=np.polyfit(right_y,right_x,2)
    return left_fit, right_fit, left_lane_indx, right_lane_indx, window_data, left_px, right_px

def previous_window_search(raw_binary_image,previous_left_fit,previous_right_fit,margin=40):
    nonzero_pixels=raw_binary_image.nonzero()
    nonzero_pixels_x=np.array(nonzero_pixels[1])
    nonzero_pixels_y=np.array(nonzero_pixels[0])
    px_y=nonzero_pixels_y
    
    l_a,l_b,l_c=previous_left_fit
    r_a,r_b,r_c=previous_right_fit
    
    left_lane_indx=((nonzero_pixels_x>(l_a*(px_y**2)+l_b*px_y+l_c-margin)) & (nonzero_pixels_x<(l_a*(px_y**2)+l_b*px_y+l_c+margin)))

    right_lane_indx=((nonzero_pixels_x>(r_a*(px_y**2)+r_b*px_y+r_c-margin)) & (nonzero_pixels_x<(r_a*(px_y**2)+r_b*px_y+r_c+margin)))
    
    # Extract pixel data
    left_x=nonzero_pixels_x[left_lane_indx]
    left_y=nonzero_pixels_y[left_lane_indx]
    right_x=nonzero_pixels_x[right_lane_indx]
    right_y=nonzero_pixels_y[right_lane_indx]

    # Combining pixel data
    left_px=[left_y,left_x]
    right_px=[right_y,right_x]
    
    # Polyfit a second-order polynomial to each pixel information
    left_fit=None
    right_fit=None
    if len(left_x)!=0:
        left_fit=np.polyfit(left_y,left_x,2)
    if len(right_x)!=0:
        right_fit=np.polyfit(right_y,right_x,2)
    
    return left_fit, right_fit, left_lane_indx, right_lane_indx, left_px, right_px, margin

def generate_polynomial(raw_image,fit_values):
    a,b,c=fit_values
    y_grid=np.linspace(0,raw_image.shape[0]-1,2*raw_image.shape[0])
    fit_polynomial=(a*np.power(y_grid,2))+(b*y_grid)+c
    return fit_polynomial, y_grid

def generate_left_window(output_image,window_datum):
    low_vertex=(window_datum[2],window_datum[0])
    high_vertex=(window_datum[3],window_datum[1])
    return cv2.rectangle(output_image, (low_vertex), (high_vertex), (0,255,0), 2)

def generate_right_window(output_image,window_datum):
     #window_data.append((window_y_low,window_y_high,
     #                      window_x_left_low,window_x_left_high,
     #                       window_x_right_low,window_x_right_high))
    low_vertex=(window_datum[4],window_datum[0])
    high_vertex=(window_datum[5],window_datum[1])
    return cv2.rectangle(output_image, (low_vertex), (high_vertex), (0,255,0), 2)

    
def visualize_windowfit(raw_binary_warped_image):
    raw_binary_warped_outimage=np.uint8(255*np.dstack((raw_binary_warped_image,raw_binary_warped_image,raw_binary_warped_image)))
    
    left_fit, right_fit, left_lane_indx, right_lane_indx, window_data, left_px, right_px=new_sliding_window(raw_binary_warped_image)
    for window_datum in window_data:
        plt.imshow(generate_left_window(raw_binary_warped_image,window_datum))
        plt.imshow(generate_right_window(raw_binary_warped_image,window_datum))
    raw_binary_warped_outimage[left_px]=[0,0,255]
    raw_binary_warped_outimage[right_px]=[255,0,0]
    plt.imshow(raw_binary_warped_outimage)
    
    leftfit_polynomial, y_grid=generate_polynomial(raw_binary_warped_image,left_fit)
    rightfit_polynomial, y_grid=generate_polynomial(raw_binary_warped_image,right_fit)
    
    plt.plot(leftfit_polynomial,y_grid,color='yellow')
    plt.plot(rightfit_polynomial,y_grid,color='yellow')
    
    plt.xlim(0, 1280)
    plt.ylim(720, 0)

def combined_windowfit_filled(raw_binary_warped_image):
    raw_binary_warped_outimage=np.uint8(255*np.dstack((raw_binary_warped_image,raw_binary_warped_image,raw_binary_warped_image)))
    
    previous_left_fit, previous_right_fit, left_lane_indx, right_lane_indx, window_data, left_px, right_px=new_sliding_window(raw_binary_warped_image)
    
    left_fit, right_fit, left_lane_indx, right_lane_indx, left_px, right_px, margin=previous_window_search(raw_binary_warped_image,previous_left_fit,previous_right_fit)
    
    leftfit_polynomial, y_grid=generate_polynomial(raw_binary_warped_image,left_fit)
    rightfit_polynomial, y_grid=generate_polynomial(raw_binary_warped_image,right_fit)
    
    window_image=np.zeros_like(raw_binary_warped_outimage)
    
    # Fill in right/left lane pixels
    raw_binary_warped_outimage[left_px[0],left_px[1]]=[255,0,0]
    raw_binary_warped_outimage[right_px[0],right_px[1]]=[255,0,0]
    
    # Generate a polygon, and recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1=np.array([np.transpose(np.vstack([(leftfit_polynomial-margin),y_grid]))])
    left_line_window2=np.array([np.flipud(np.transpose(np.vstack([(leftfit_polynomial+margin),y_grid])))])
    left_line_points=np.hstack((left_line_window1,left_line_window2))
    
    right_line_window1=np.array([np.transpose(np.vstack([(rightfit_polynomial-margin),y_grid]))])
    right_line_window2=np.array([np.flipud(np.transpose(np.vstack([(rightfit_polynomial+margin),y_grid])))])
    right_line_points=np.hstack((right_line_window1,right_line_window2))
    
    # Draw filled rectangle windows onto a warped image
    cv2.fillPoly(window_image,np.int_([left_line_points]),(0,255,0))
    cv2.fillPoly(window_image,np.int_([right_line_points]),(0,255,0))
    combined=cv2.addWeighted(raw_binary_warped_outimage,1., window_image, 0.3, 0)
    plt.imshow(combined)
     
    # Plot fitted polynomial onto the warped image
    plt.plot(leftfit_polynomial, y_grid, color='yellow')
    plt.plot(rightfit_polynomial, y_grid, color='yellow')
    
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    
    return left_fit, right_fit, left_px, right_px