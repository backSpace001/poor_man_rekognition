import cv2
import label_image

size = 4


# load the xml file
classifier = cv2.CascadeClassifier('F:/dd/Library/etc/haarcascades/haarcascade_frontalface_alt.xml')

im = cv2.imread('C:/Users/Faiz Khan/Desktop/ddd/facial expression/test/3.jpg', 0 )
#im=cv2.flip(im,1,0) #Flip to act as a mirror

# Resize the image to speed up detection
mini = cv2.resize(im, (int(im.shape[1]/size), int(im.shape[0]/size)))

# detect MultiScale / faces 
faces = classifier.detectMultiScale(mini)

# Draw rectangles around each face
for f in faces:
    (x, y, w, h) = [v * size for v in f] #Scale the shapesize backup
    cv2.rectangle(im, (x,y), (x+w,y+h), (0,255,0), 4)
        
    #Save just the rectangle faces in SubRecFaces
    sub_face = im[y:y+h, x:x+w]

    FaceFileName = "test.jpg" #Saving the current image for testing.
    cv2.imwrite(FaceFileName, sub_face)
        
    text = label_image.main(FaceFileName)# Getting the Result from the label_image file, i.e., Classification Result.
    text = text.title()# Title Case looks Stunning.
    font = cv2.FONT_HERSHEY_TRIPLEX
    cv2.putText(im, text,(x,y), font, 1, (255,0,0), 2)
 # Show the image
cv2.imshow('Capture',   im)
key = cv2.waitKey(10000)
