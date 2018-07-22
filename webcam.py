import cv2 as cv

def display():
    cap = cv.VideoCapture(0)

    while (True):
        ret, frame = cap.read()
        img = cv.cvtColor(frame, cv.COLOR_BGR2BGRA)

        cv.imshow('Frame', img)

        if cv.waitKey(1) & 0xFF == ord('q'):
            # out = cv.imwrite('capture.jpg', frame)
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__": display()
