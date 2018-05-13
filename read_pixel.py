import cv2
import numpy as np
import scipy.misc
import moviepy.editor as mpy # Clean. Then use mpy.VideoClip, etc.

#Einstellung
cap = cv2.VideoCapture('/Users/matthew/sciebo/Projekt/bird_detector/assets/cctv_coba.m4v')
fgbg = cv2.createBackgroundSubtractorKNN()

#configurations
delay_time = 60
seconds = 60 #seconds used to track
fps = 15
per_frame = fps # frames per count
######

# Define the codec and create VideoWriter object
# fourcc = VideoWriter_fourcc('M','J','P','G')
# out = VideoWriter('output.m4v',fourcc, 20.0, (352*2,288*2), True)

pict_arr = [[{}]]
# pict_arr_y = np.array([[{}]])

height = 0
# count_frame = 0

def argmax_dict (x) :
    # A function to take the highest argument
    x1 = list(x.keys())[0]
    for x2 in x.keys():
        if(x[x1]<x[x2]):
            x1=x2
    return x1

def make_background(pict_arr): # output frames
    result=[[]]
    for x in range(height):
        try: # to check the existance of the index
            (result[x])
        except IndexError as e:
            result.append([{}])

        for y in range(width):
            try: # to check the existance of the index
                (result[x][y])
            except IndexError as e:
                result[x].append({})
            # print(pict_arr[x][y])
            i = argmax_dict(pict_arr[x][y])
            # print(i)
            result[x][y] = i
    result_np = np.array(result, dtype=np.uint8)
    return result_np


# the main things are here
images_list=[]
for count_frame in range(seconds):
    count_frame = count_frame*per_frame
    print(count_frame)
    cap.set(1,count_frame);
    ret, frame = cap.read()

    if not height:
        height, width = frame.shape[:2]
        # print(height)
    fgmask = fgbg.apply(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) # Convert original frame to grayscale
    # out.write(fgmask)
    # scipy.misc.toimage(frame, cmin=0.0, cmax=...).save('log_background/'+str(order)+'.jpg')

    # let x and y // remember both are inverted x in respect to height
    for x in range(height):
        # print('x = ',x)
        try: # to check the existance of the index
            (pict_arr[x])
        except IndexError as e:
            pict_arr.append([{}])


        for y in range(width):
            try: # to check the existance of the index
                (pict_arr[x][y])
            except IndexError as e:
                pict_arr[x].append({})


            # print('y = ',y)
            t_color= frame[x,y] # t_color is the color of the pixel of that specific time

            # pict_arr[x,y] = np.array([])
            # pict_arr_y[x,y] = np.array([])
            # print('t_color = ',t_color)
            # n = np.argwhere(pict_arr[x,y]==x_find)
            try:
                (pict_arr[x][y][t_color])
            except KeyError as e:
                pict_arr[x][y][t_color]=0
            else:
                pict_arr[x][y][t_color]+=1


            # cv2.waitKey(1)

        cv2.waitKey(1)
        # cv2.waitKey(1)
    # print('pict_arr ',count,' = ',pict_arr)
    cv2.imshow('frame',frame)
    result_np = make_background(pict_arr)
    cvt_result_np=cv2.cvtColor(result_np,cv2.COLOR_GRAY2RGB)
    images_list.append(cvt_result_np)
    cv2.imshow('result', result_np)
    k = cv2.waitKey(10) & 0xff
    if k == ord('q'):
        break

# print(images_list)

clip = mpy.ImageSequenceClip(images_list, fps=int(2), with_mask=False)
clip.write_videofile("log_background/out_background.mp4")
scipy.misc.toimage(result_np, cmin=0.0, cmax=...).save('log_background/out_background.jpg')

cv2.waitKey(0)

cap.release()
# out.release()
cv2.destroyAllWindows()
