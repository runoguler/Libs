import cv2 as cv

def main():
    face_cascade = cv.CascadeClassifier('/Users/onur/anaconda3/envs/main/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
    face_profile_cascade = cv.CascadeClassifier('/Users/onur/anaconda3/envs/main/share/OpenCV/haarcascades/haarcascade_profileface.xml')
    body_cascade = cv.CascadeClassifier('/Users/onur/anaconda3/envs/main/share/OpenCV/haarcascades/haarcascade_fullbody.xml')
    smile_cascade = cv.CascadeClassifier('/Users/onur/anaconda3/envs/main/share/OpenCV/haarcascades/haarcascade_smile.xml')

    cap = cv.VideoCapture(0)

    while (True):
        ret, img = cap.read()
        if(not ret): continue

        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)
        faces_profile = face_profile_cascade.detectMultiScale(gray_img, 1.1, 3)
        bodies = body_cascade.detectMultiScale(gray_img, 1.1, 4)

        for (x, y, w, h) in faces:
            img = cv.rectangle(img, (x, y), (x+w,y+h), (0, 0, 255))
        for (x, y, w, h) in faces_profile:
            img = cv.rectangle(img, (x, y), (x+w,y+h), (0, 255, 0))
        for (x, y, w, h) in bodies:
            img = cv.rectangle(img, (x, y), (x+w,y+h), (255, 0, 0))

        cv.imshow('frame', img)

        if cv.waitKey(1) & 0xFF == ord('q'):
            # out = cv.imwrite('capture.jpg', frame)
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__": main()
