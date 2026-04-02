import cv2
import HandTrackingModule as htm
import csv

cap = cv2.VideoCapture(0)
detector = htm.HandDetector()

label = 'T'
count = 0

file = open("sign_data.csv", "a", newline="")
writer = csv.writer(file)

paused = False  # press P to pause/resume

print("Show sign:", label)
print("Press P to pause/resume")
print("Press Ctrl+C in terminal to stop")

while True:
    try:
        success, img = cap.read()
        if not success:
            print("Camera not working")
            break

        img = cv2.flip(img, 1)  # flip horizontally

        img = detector.findHands(img)
        lmList = detector.findPosition(img, Draw=False)

        data = None

        if len(lmList) == 21:

            # taking x and y(height,width) co-ordinates
            xs = [lm[1] for lm in lmList]
            ys = [lm[2] for lm in lmList]

            # calculating the hand size(height and width)
            width = max(xs) - min(xs)
            height = max(ys) - min(ys)
            scale = max(width, height)
            
            # Taking the full hand
            if width != 0 and height != 0:

                # Taking wrist as it is stable when compared to other co-ordinates
                wrist_x = lmList[0][1]
                wrist_y = lmList[0][2]

                data = []

                for lm in lmList:
                    norm_x = (lm[1] - wrist_x) / width
                    norm_y = (lm[2] - wrist_y) / height
                    data.append(norm_x)
                    data.append(norm_y)

                data.append(label)

                # save automatically if not paused
                if not paused:
                    writer.writerow(data)
                    count += 1

        # show image
        cv2.imshow("Image", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('p'):
            paused = not paused
            print("Paused" if paused else "Resumed")

    except KeyboardInterrupt:
        print("\nStopped by user (Ctrl+C)")
        break

file.close()
cap.release()
cv2.destroyAllWindows()

print("Total samples saved:", count)
