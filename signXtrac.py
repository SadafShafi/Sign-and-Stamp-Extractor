import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
from ultralytics import YOLO


class signXtrac:
    def __init__(self):
        self.model = YOLO('yolov8n.pt') 
        self.model = YOLO('best.pt') 

    def textDisappear(self,image):
        image = image[2*image.shape[0]//3:,:,:]    # uncomment for taking last 2/3rd of the image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        # Find contours
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Loop through contours
        for contour in contours:
            # Get the bounding box of the contour
            x, y, w, h = cv2.boundingRect(contour)
            # Compute the aspect ratio of the bounding box
            aspect_ratio = w / float(h)
            # Filter out contours that are too wide or too tall to be text
            if aspect_ratio > 5 or aspect_ratio < 0.2:
                continue
            # Get the region of interest (ROI) corresponding to the contour
            roi = image[y:y+h, x:x+w]
            # Compute the average intensity of the ROI
            avg_intensity = np.mean(roi)
            # If the average intensity is below a threshold, assume the ROI contains readable text
            if avg_intensity < 200:
                # Fill the ROI with white color
                cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 255), -1)
        return image


    def extracSign(self,impath):

        try:

          imageTexted = cv2.imread(impath)    # image with text in the background
          image = self.textDisappear(imageTexted.copy()) 
          result = self.model(image)
          boxes = result[0].boxes.xyxy.tolist()

          cropped_images = []
          for bbox in boxes:
              xmin, ymin, xmax, ymax = map(int, bbox)
              # print(xmin, ymin, xmax, ymax)
              # xmin -= xmin//3
              # ymin -= ymin//2
              # xmax += xmax//3
              # ymax += ymax//3
              # print(xmin, ymin, xmax, ymax)
              imageTexted = imageTexted[2*imageTexted.shape[0]//3:,:,:] 
              cropped_image = imageTexted[ymin:ymax, xmin:xmax]
              cropped_image = self.textDisappear(cropped_image)
              cropped_images.append(cropped_image)
          return cropped_images
        except Exception as e:
          print(e)
          return e









##################33   Test Here 

signIns = signXtrac()
result = signIns.extracSign('sampleimg2.png')
print(len(result))
cv2.imshow('image',result[1])
cv2.waitKey(0)
# and finally destroy/close all open windows
cv2.destroyAllWindows()



