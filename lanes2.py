import cv2
import numpy as np
import matplotlib.pyplot as plt
def make_coordiantes(image,line_paramters):
    slope,intercept=line_paramters
    y1=image.shape[0]
    y2=int(y1*(3/5))
    x1=int((y1-intercept)/slope)
    x2=int((y2-intercept)/slope)

    return np.array([x1,y1,x2,y2])

def averaged_line(image,lines):
    left_fit=[]
    right_fit=[]
    for line in lines:
        x1,y1,x2,y2=line.reshape(4)
        parameter=np.polyfit((x1,x2),(y1,y2),1)
        slope=parameter[0]
        intercept=parameter[1]
        if slope < 0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))
    left_fit_average=np.average(left_fit,axis=0)
    right_fit_average=np.average(right_fit,axis=0)
    left_line=make_coordiantes(image,left_fit_average)
    right_line=make_coordiantes(image,right_fit_average)

    return np.array([left_line,right_line])

def canny(image):
    gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    blur=cv2.GaussianBlur(gray,(5,5),0)
    canny=cv2.Canny(blur,50,150)
    return canny

def display_lines(image,lines):
    line_image=np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2=line.reshape(4)
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
    return line_image

def region_of_interest(image):
    height=image.shape[0]
    polygons=np.array([[(200,height),(1100,height),(550,250)]])
    mask=np.zeros_like(image)
    cv2.fillPoly(mask,polygons,255)
    masked_image=cv2.bitwise_and(image,mask)
    return masked_image

"""
image=cv2.imread("test_image.jpeg")
lane_image=np.copy(image)
canny=canny(lane_image)
masked_image=region_of_interest(canny)
lines=cv2.HoughLinesP(masked_image,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
averaged_line=averaged_line(image,lines)
line_image=display_lines(lane_image,averaged_line)
final_image=cv2.addWeighted(lane_image,0.8,line_image,1,1)
"""
cap=cv2.VideoCapture("test2.mp4")
while(cap.isOpened()):
    _,frame =cap.read()
    canny_image=canny(frame)
    masked_image=region_of_interest(canny_image)
    lines=cv2.HoughLinesP(masked_image,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
    averaged_lines=averaged_line(frame,lines)
    line_image=display_lines(frame,averaged_lines)
    final_image=cv2.addWeighted(frame,0.8,line_image,1,1)
    #cv2.imshow("Test1",canny)
    #cv2.imshow("Test2",line_image)
    cv2.imshow("Test3",final_image)
    if(cv2.waitKey(1)==ord("q")):
        break

cap.release()
cv2.destroyAllWindows()
