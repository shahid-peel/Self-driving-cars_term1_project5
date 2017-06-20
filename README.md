## Writeup Template

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.


[//]: # (Image References)
[image0]: ./output_images/car_and_not_car_lot.jpg
[image0b]: ./output_images/HOG_example.jpg
[image0c]: ./output_images/windows.jpg
[image1a]: ./output_images/raw_plot1.jpg
[image2a]: ./output_images/raw_plot2.jpg
[image3a]: ./output_images/raw_plot3.jpg
[image4a]: ./output_images/raw_plot4.jpg
[image5a]: ./output_images/raw_plot5.jpg
[image6a]: ./output_images/raw_plot6.jpg
[image1b]: ./output_images/plot1.jpg
[image2b]: ./output_images/plot2.jpg
[image3b]: ./output_images/plot3.jpg
[image4b]: ./output_images/plot4.jpg
[image5b]: ./output_images/plot5.jpg
[image6b]: ./output_images/plot6.jpg
[video1]: ./project_video.mp4


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
## Writeup / README

### 1. Writeup / README that includes all the rubric points and how you addressed each one.  

You're reading it!

## Histogram of Oriented Gradients (HOG)

### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the 1st code cell of the IPython notebook.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image0]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image0b]

### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters. First I experimented with the different color spaces (RGB, HSV, LUV, HLS, YUV, YCrCb) and found that YCrCb gave me the best results. Next I tried to determine which (or all) of the features gave me the best result and it turned out that the combination of all of them was the best - get features for the different channels and concatenate them together. 

For HOG orientations, I wanted to capture gradients better so I tried to go down to 20 degree bins (18 bins out of 360 degrees). This increased the number of features but also resulted in a lot more false positives. In the end I used 9 orientations. Similarly I experimented with number of pixels in the cell and at 16x16 it gave me decent results while keeping the feature vector size low.

The upside of using only 9 orientations (40 degrees) was that the feature vector stayed relatively small and therefore was faster to train.


### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for this step is contained in the 2nd code cell of the IPython notebook.  My choice of classifier was a mltilayer perceptron (MLP). But before I trained the classifier I first randomized the samples. Used a training and test split of 80-20. Before I fed the features to the classifier, I normalized the feature vector. I used the same scaler for both training and inference time. 

## Sliding Window Search

### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code for this step is contained in the 1st and 2nd code cell of the IPython notebook. The utility function `slide_window` is in the 1st cell and finds out the extents of all the windows given windowing parameters. The `slide_window` function is called by `detect_car_and_draw_bounding_box` (2nd code cell). This is where I created windows of different sizes. 

I ignored the top half of hte image as that is above the horizon. For the 2nd half of the image, I used small windows on the pixels closer to the horizon and larger windows for pixels closer to the bottom of the image.  This way cars closers up are detected by the bigger windows while the smaller cars at the distance are detected by small ones.

Another important factor here is the overlap ratio. For windows closer up, the same overlap ratio of 0.5 would have resulted in a gap between two detectios e.g. for a 128x128 window, that gap would be 64 pixels. That was too big of a hole where I was not getting detections and so I increased the overlap ratio for larger windows.

### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

In the end I used the YCrCb 3-channel HOG features as they provided the best results. Here are some example images:

![alt text][image0c]
---

## Video Implementation

### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_video.mp4)


### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The code for this step is contained in the 1st and 1st code cell of the IPython notebook. The `draw_boxes` function implements a running heatmap. Every detected car window adds heat to the heatmap. Overlapping window regions therefore get higher probability of car presence. There is thresholding implemneted as well which removes false positives.

To remove false positive, I used a deque to maintain the latest 10 heatmap readings. Then I summed up the heatmaps to get the liklihood of car presence in the last 10 frames. By thresholding (at 4) i ensure that the false positives are removed.  The number 4 seems high but given the fact that my classifier was giving multiple windows on the cars, it works out pretty well.

```python
heat = np.zeros_like(img[:,:,0]).astype(np.float)
heatmap = add_heat(heat, bboxes)
heatmaps.append(heatmap)
combined = sum(heatmaps)
heatmap = apply_threshold(combined, 4)
```

Once I have a heatmap that represents the running position of the cars, I used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

## Here are six frames and their corresponding raw heatmaps:

![alt text][image1a]
![alt text][image2a]
![alt text][image3a]
![alt text][image4a]
![alt text][image5a]
![alt text][image6a]

## Here are the same six frames and their corresponding integrated heatmap:

Note that the randomization seed was kept the same to allow for comparison of results. The images below have a running heat map with cooling and thresholding.

![alt text][image1b]
![alt text][image2b]
![alt text][image3b]
![alt text][image4b]
![alt text][image5b]
![alt text][image6b]


---

## Discussion

### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Issues:
- Biggest issue overall was trainig time. My implementation is not the most efficient as I have focussed more on filteration of results.

Where pipeline will likely fail:
- If the vehicles always gets detected by a single window, then it will be filtered out as a false positive. For the project video it doesn't happen as much but in another scanario it might. This is compensated by having windows of multiple sizes in the search, which gives us multiple windows per car.  

How to make the solution more robust:
- I would make the solution better by using HOG feature more intelligently for test images.
- Another way of making the solution more robust is to filter out near duplicates from training data. Because some of the images are from a series of frames in a video, some of them might end up in training while their (very similar) neighbors would end up in the testing. This almost breaks the whole notion of never testing of data that you have already seen.
