import cv2 as cv
from matplotlib import pyplot as pyt

img = cv.imread("image2.jpeg")

img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)



stop_data = cv.CascadeClassifier('stop_data.xml')
  
found = stop_data.detectMultiScale(img_gray, 
                                   minSize =(20, 20))
amount_found = len(found)
  
if amount_found != 0:
      
   
    for (x, y, width, height) in found:
        
        
        cv.rectangle(img_rgb, (x, y), 
                      (x + height, y + width), 
                      (0, 255, 0), 5)

pyt.subplot(1,1,1)
pyt.imshow(img_rgb)
pyt.show()
