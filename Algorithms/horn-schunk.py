import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.filters import convolve as filter2

FILTER = 7
QUIVER = 10

def HS(im1, im2, alpha, Niter):
    # Set up initial velocities
    uInitial = np.zeros([im1.shape[0], im1.shape[1]])
    vInitial = np.zeros([im1.shape[0], im1.shape[1]])
    U = uInitial
    V = vInitial

    # Estimate derivatives
    [fx, fy, ft] = computeDerivatives(im1, im2)

    # Averaging kernel
    kernel = np.array([[1/12, 1/6, 1/12],
                       [1/6, 0, 1/6],
                       [1/12, 1/6, 1/12]], float)

    for _ in range(Niter):
        uAvg = filter2(U, kernel)
        vAvg = filter2(V, kernel)
        der = (fx * uAvg + fy * vAvg + ft) / (alpha**2 + fx**2 + fy**2)
        U = uAvg - fx * der
        V = vAvg - fy * der

    return U, V

def computeDerivatives(im1, im2):
    kernelX = np.array([[-1, 1],
                        [-1, 1]]) * .25
    kernelY = np.array([[-1, -1],
                        [1, 1]]) * .25
    kernelT = np.ones((2, 2)) * .25

    fx = filter2(im1, kernelX) + filter2(im2, kernelX)
    fy = filter2(im1, kernelY) + filter2(im2, kernelY)
    ft = filter2(im1, kernelT) + filter2(im2, -kernelT)

    return fx, fy, ft

def compareGraphs(u, v, Inew, scale=3):
    result = Inew.copy()
    height, width = result.shape[:2]
    for i in range(0, height):
        for j in range(0, width):
            if (i % QUIVER == 0) and (j % QUIVER == 0):
                cv2.arrowedLine(result, (j, i), (int(j + v[i, j] * scale), int(i + u[i, j] * scale)), (0, 0, 255), 1)
    return result

def demo(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255

    height, width, _ = frame1.shape
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (width, height))

    while True:
        ret, frame2 = cap.read()
        if not ret:
            break
        next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        Iold = gaussian_filter(prvs.astype(float), FILTER)
        Inew = gaussian_filter(next_frame.astype(float), FILTER)

        U, V = HS(Iold, Inew, 1, 100)
        result_frame = compareGraphs(U, V, frame2, scale=3)
        out.write(result_frame)

        cv2.imshow('Result', result_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        prvs = next_frame

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    video_path = r'Video/running.mp4'
    demo(video_path)
