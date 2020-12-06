import cvlib as cv
import cv2
from PIL import Image

cap = cv2.VideoCapture(0)

font_scale=1
thickness = 2
red = (0,0,255)
green = (0,255,0)
blue = (255,0,0)
font=cv2.FONT_HERSHEY_SIMPLEX

#File must be downloaded
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


while(cap.isOpened()):
    ret, frame = cap.read()
    print("1")
    if ret == True:
        print("2")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.4, 4)
        print("21")
        print(faces)
        for (x, y, w, h) in faces:
            print("3")
            cv2.rectangle(frame, (x, y), (x+w, y+h), blue, 2)
            
            croped_img = frame[y:y+h, x:x+w]
            pil_image = Image.fromarray(croped_img, mode = "RGB")
            pil_image = train_transforms(pil_image)
            image = pil_image.unsqueeze(0)
            
            
            result = loaded_model(image)
            _, maximum = torch.max(result.data, 1)
            prediction = maximum.item()

            print("4")
            if prediction == 0:
                cv2.putText(frame, "Masked", (x,y - 10), font, font_scale, green, thickness)
                cv2.rectangle(frame, (x, y), (x+w, y+h), green, 2)
                print("5")
            elif prediction == 1:
                cv2.putText(frame, "No Mask", (x,y - 10), font, font_scale, red, thickness)
                cv2.rectangle(frame, (x, y), (x+w, y+h), red, 2)
                print("6")
        print("22")
        cv2.startWindowThread()
        cv2.imshow('frame',frame)
        print("7")
        
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            print("8")
            break
    else:
        print("9")
        break
print("10")
cap.release()
cv2.destroyAllWindows()

