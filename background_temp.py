#Note: in here, I put all the iamge processing...

import cv2
import numpy as np
import scipy.misc


#Einstellung
cap = cv2.VideoCapture('/Users/matthew/sciebo/Projekt/bird_detector/assets/cctv_coba.m4v')

mask_ori = cv2.imread('log_background/out_background.jpg', 0)
delay_time = 60

# Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
# out = cv2.VideoWriter('output.m4v',fourcc, 20.0, (352*2,288*2), True)

order=0  # Order of log image for background substraction

# All algorithm necessary
kernel_open = np.ones((1,1),np.uint8)
kernel_close = np.ones((2,2),np.uint8)
fgbg = cv2.createBackgroundSubtractorKNN()

while(1):
    ret, ori = cap.read()

    #Set the ROI
    height, width = ori.shape[:2]
    height /= 1.2
    width /= 2
    height = int(height)
    width = int(width)
    frame = ori[0:height, 0:width]
    mask = mask_ori[0:height, 0:width]
    frame_oris = frame

    cv2.imshow('original', frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    knn_mask = fgbg.apply(frame)
    print(frame[0,0]," ",mask[0,0])
    cv2.imshow('frame',frame)
    frame=frame.astype(np.int16)
    mask=mask.astype(np.int16)

    opening = cv2.morphologyEx(knn_mask, cv2.MORPH_OPEN, kernel_open)
    # erosion = cv2.erode(knn_mask,kernel,iterations = 1)
    fgmask = np.absolute(frame-mask) # the difference of both of the frame
    fgmask = fgmask.astype(np.uint8)
    print(fgmask[0,0])

    # thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1] # for the
    # scipy.misc.toimage(fgmask, cmin=0.0, cmax=...).save('log_background/'+str(order)+'.jpg')

    # fgmask = np.power(fgmask, 2)

    combine = opening+fgmask

    # opening_opening = cv2.morphologyEx(opening_raw, cv2.MORPH_OPEN, kernel_open)
    # combine = cv2.morphologyEx(opening_opening, cv2.MORPH_CLOSE, kernel_close)

    ret,thresh1 = cv2.threshold(combine,50,255,cv2.THRESH_BINARY) # for the fgmask or the delta of the frame
    #detect the contour
    # print(combine)
    try:
        image,cnts, _ = cv2.findContours(thresh1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    except ValueError as e:
        print('Error')
        continue

    cv2.waitKey(1)
    print('h')

    frame=frame.astype(np.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

    i=0
    font = cv2.FONT_HERSHEY_SIMPLEX

    frame_draw=frame #Save location to draw the frame
    for c in cnts:

		# if the contour is too small, ignore it
		# if cv2.contourArea(c) < args["min_area"]:
		# 	continue

		# compute the bounding box for the contour, draw it on the frame,
		# and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame_draw, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(frame,str(i),(x,y), font, 0.5,(50,0,255))
        i+=1

    combine = cv2.cvtColor(combine, cv2.COLOR_GRAY2RGB)
    cv2.imshow('thresh1',thresh1)
    cv2.imshow('fgmask',fgmask)
    # cv2.imshow('knn_mask',opening)
    # cv2.imshow('combine',opening_raw)
    cv2.imshow('frame', frame)

    scipy.misc.toimage(frame, cmin=0.0, cmax=...).save('log_background/'+str(order)+'.jpg')
    scipy.misc.toimage(frame_oris, cmin=0.0, cmax=...).save('log_background_ori/'+str(order)+'.jpg')
    # cv2.imshow('mask',mask)
    order+=1
    k = cv2.waitKey(30) & 0xff
    if k == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
