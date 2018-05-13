import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('log_background/out_background.jpg',0)
plt.hist(img.ravel(),256,[0,256]); plt.show()

equ = cv2.equalizeHist(img)
# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
equ = clahe.apply(img)

res = np.hstack((img,equ)) #stacking images side-by-side
cv2.imshow('res', res)
cv2.imwrite('res.png',res)

k = cv2.waitKey(0)

cap.release()
out.release()
cv2.destroyAllWindows()
