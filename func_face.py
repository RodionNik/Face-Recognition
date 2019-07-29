import numpy as np
import cv2
import face_recognition
import pickle

conf_thr=0.4 #по умолчанию

data = pickle.loads(open('encodings.pickle', "rb").read())
net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt","res10_300x300_ssd_iter_140000.caffemodel")

def face_detect_dl(image):
    
    #bug=0
    #startX=0
    #startY=0
    #endX=0
    #endY=0
    lst=[]
    j=0
    
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    
    for i in range(0,detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        #print (i)
        if confidence > conf_thr:
            #bug=1
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            lst.append((startY, startX,  endY,  endX))
            j+=1
            #text = "{:.2f}%".format(confidence * 100)
            #y = startY - 10 if startY - 10 > 10 else startY + 10
            
            #cv2.rectangle(image, (startX, startY), (endX, endY),(0, 0, 255), 2)
            #cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            
    return  lst  #startY, startX,  endY,  endX)

def recgnize(image, boxes, tol ):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []
    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"],
                                                 encoding,tolerance=tol)
        name = "Unknown"
        
        if True in matches:
            #print(matches,'\n')
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            #print(matchedIdxs)
            counts = {}
            for i in matchedIdxs:
                name = data["names"][i]
                #print(name)
                counts[name] = counts.get(name, 0)+1
            name = max(counts, key=counts.get)
        #print(counts)
        names.append(name)
    
    return names