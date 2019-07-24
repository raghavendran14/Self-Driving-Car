**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


In this project i used two files , one for image and another one for video

### Camera Calibration
 I used the cv2.imread(fname) to read the image, and to convert into the gray scale image i used  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 i.e COLOR_BGR2GRAY.
 
 I calculated the image points and the object points first,then  i used these points to compute the camera caliberation .
 X ,Y,Z are the coordinators of the chess board.
 
 Finding chessboard corners
 cv2.findChessboardCorners() 
 
 Drawing detected corners on an image:
 cv2.drawChessboardCorners() 
 after finding the object and image points , i used these points to compute the camera caliberation.
 
 Camera calibration, given object points, image points, and the shape of the grayscale image:
 cv2.calibrateCamera() 
 
 later i applied the distortion correction to the test image using
 cv2.undistort()
 the output is below
 
![](dst.png)

# I applied the distortion correction to the raw image

i used below function to correct distort raw image  
image = cv2.undistort(raw, mtx, dist, None, mtx)
 The out is :
 ![]('image.png')
 

# Use color transforms, gradients, etc., to create a thresholded binary image.
i called the below function to create thrshold binary image
combined_binary = apply_threshold_v2(image, xgrad_thresh=xgrad_thresh_temp, s_thresh=s_thresh_temp)

![]('combined_binary.png')

#
# Apply a perspective transform to rectify binary image ("birds-eye view")
now its time to Apply a perspective transform to rectify binary image,  
src = np.float32(
    [[120, 720],
     [550, 470],
     [700, 470],
     [1160, 720]])

dst = np.float32(
    [[200,720],
     [200,0],
     [1080,0],
     [1080,720]])

M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)
later i called the below function to get the result
warped1 = cv2.warpPerspective(combined_binary, M, (imshape[1], imshape[0]), flags=cv2.INTER_LINEAR)

![]('output_images/warped1')

# Detect lane pixels and fit to find the lane boundary
step 1: Histogram and get pixels in window

leftx, lefty, rightx, righty = histogram_pixels(warped, horizontal_offset=horizontal_offset)

step 2: Fit a second order polynomial to each fake lane line
left_fit, left_coeffs = fit_second_order_poly(lefty, leftx, return_coeffs=True)

step 3:Plotting data

result:
![]('output_images/warped')
![]('output_images/trace')

# Determine the curvature of the lane and vehicle position with respect to center.

step 1: Determine curvature of the lane
step 2: Define y-value where we want radius of curvature
step 3: I'll choose the maximum y-value, corresponding to the bottom of the image

# Warp the detected lane boundaries back onto the original image
Warp lane boundaries back onto original image by calling the below function.

lane_lines = cv2.warpPerspective(trace, Minv, (imshape[1], imshape[0]), flags=cv2.INTER_LINEAR)

Convert to colour
combined_img = cv2.add(lane_lines, image)

result:
![]('output_images/combined_img')

Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position
![]('output_images/combined_img')

# Pipeline_Video
I used a another file (project2_video_advanced_lane_line.ipynb), 
here a link:
https://classroom.udacity.com/nanodegrees/nd013/parts/edf28735-efc1-4b99-8fbb-ba9c432239c8/modules/5d1efbaa-27d0-4ad5-a67a-48729ccebd9c/lessons/7cb63828-36aa-4cea-9239-700b5ea41f0b/concepts/0a96d23f-6c22-4053-a7f6-83e12ce5a6ec

## Discussion

I did a lot of experiments on this project to keep track of the lane in video(i concentrated mainly on what attributs should i keep for each line and when i should consider data from previous frames and when i use new data ).
I have also tried with diffrent source pts for the perspective transform process.a small variation had a huge imapct on the final result . i could try to get multiple source and destination pts and took the average to have a better  matrix.

My pipeline fails in the challenge video, it did not identify the lane lines.Improved image processing using color would certainly improve it.
knowing the left lane is yellow and continuous and the middle one is white and dotted is something i did not take advantage.
in my harder challenge video, the curve are too closed, 
video did not always record both lines.
using the one lane line to infer the position of the other lane is somethimg which i could apply to the pipeline to make it more robust for the harder challenge.


