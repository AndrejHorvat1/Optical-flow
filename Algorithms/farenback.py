import cv2
import numpy as np

capture = cv2.VideoCapture(r'Video/bullet.mp4')

_, frame1 = capture.read()
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
hsv_mask = np.zeros_like(frame1)
hsv_mask[..., 1] = 255

while(1):

	_, frame2 = capture.read()
	next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

	flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

	mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

	hsv_mask[..., 0] = ang * 180 / np.pi / 2

	hsv_mask[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

	rgb_representation = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)

	cv2.imshow('frame2', rgb_representation)
	kk = cv2.waitKey(1) & 0xff

	if kk == ord('e'):
		break
	elif kk == ord('s'):
		cv2.imwrite('Optical_image.png', frame2)
		cv2.imwrite('HSV_converted_image.png', rgb_representation)
	prvs = next

capture.release()
cv2.destroyAllWindows()



