import cv2
import numpy as np

video = cv2.VideoCapture(r'Video/running.mp4')

maxCorners = 30
qualityLevel = 0.3
minDistance = 7
blockSize = 7
winSize = (15, 15)  
maxLevel = 2
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
color = np.random.randint(0, 255, (maxCorners, 3))

while True:
    ret, frame = video.read()
    if not ret:
        break

    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if 'oldGray' not in locals():
        oldGray = frameGray.copy()
        p0 = cv2.goodFeaturesToTrack(oldGray, maxCorners, qualityLevel, minDistance, blockSize=blockSize)
        mask = np.zeros_like(frame)
        continue

    p1, st, err = cv2.calcOpticalFlowPyrLK(oldGray, frameGray, p0, None, winSize=(15, 15), maxLevel=maxLevel, criteria=criteria)
    goodNew = p1[st == 1]
    goodOld = p0[st == 1]

    for i, (new, old) in enumerate(zip(goodNew, goodOld)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)

    output = cv2.add(frame, mask)

    cv2.imshow('Optical Flow', output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    oldGray = frameGray.copy()
    p0 = cv2.goodFeaturesToTrack(oldGray, maxCorners, qualityLevel, minDistance, blockSize=blockSize)

video.release()
cv2.destroyAllWindows()
