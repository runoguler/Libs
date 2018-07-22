import numpy as np
import cv2 as cv

def main():
    face_cascade = cv.CascadeClassifier('/Users/onur/anaconda3/envs/main/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')

    cap = cv.VideoCapture(0)

    while (True):
        ret, frame = cap.read()
        if(not ret): continue

        gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)

        for (x, y, w, h) in faces:
            print((x, y, w, h))
            tracking = 1
            if(tracking):
                track_window = (x, y, w, h)
                while (True):
                    roi = frame[y:y + h, x:x + w]
                    hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
                    mask = cv.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
                    roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
                    cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)

                    term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)

                    while (1):
                        ret, frame = cap.read()

                        if ret == True:
                            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
                            dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

                            ret, track_window = cv.CamShift(dst, track_window, term_crit)

                            pts = cv.boxPoints(ret)
                            pts = np.int0(pts)
                            img2 = cv.polylines(frame, [pts], True, 255, 2)
                            cv.imshow('img2', img2)

                            if cv.waitKey(1) & 0xFF == ord('q'):
                                exit(0)

                        else:
                            break
            else:
                frame = cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0))

        cv.imshow('frame', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__": main()
