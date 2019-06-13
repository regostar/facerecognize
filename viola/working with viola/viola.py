'''  
viola jones algo

'''

import cv2

def facechop(image):  
    facedata = "haarcascade_frontalface_alt.xml"
    cascade = cv2.CascadeClassifier(facedata)

    img = cv2.imread(image)

    #minisize = (img.shape[1],img.shape[0])
    #miniframe = cv2.resize(img, minisize)

    faces = cascade.detectMultiScale(img)

    for f in faces:
        x, y, w, h = [ v for v in f ]
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255))
        
        sub_face = img[y:y+h, x:x+w]
        height, width = sub_face.shape[:2]
        dst = cv2.resize(sub_face, (2*width, 2*height), interpolation = cv2.INTER_CUBIC)
        face_file_name ="face_"+str(y) + "_.jpg"
        cv2.imwrite(face_file_name, dst)

    cv2.imshow(image, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return

facechop("31.jpg")
