#seaborn
# import seaborn as sns
# import matplotlib.pyplot as plt

# plt.title()
# data=sns.dataset("iris")
# sns.set_theme(data,style='',palette='')
# sns.scatterplot(data,x='',y='',hue='',style='')
# sns.lmplot(data,x='',y='',hue='',heigt=4)
# sns.set_palette('')
# #sns.histplot(data,x='',y='',hue='',kde=True)
# sns.boxplot(data,x='',y='')
# #sns.violinplot(data,x='',y='',inner='quart')
# sns.pairplot(data,hue='')
# g=sns.FaceGrid(data,col='',row='',height=4,aspect=1.0)
# g=map.dataframe(sns.histplot,x='',hue='',kde='')
# g=set.titles(col_templete='',row_templete='')
# g.set(xlim=(),ylim=())

# plt.show()

#opencv
# import cv2
# image = cv2.imread("C:/rokey/python/ch21/task/sample2.jpg")
# resized = cv2.resize(image,(500,500))
# gray=cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)

# (h,w)=image.shape[:2]
# center=(w//2,h//2)
# #m=cv2.getRotationMatrix2D(center,45,1.0)
# #rotated=cv2.warpAffine(image,m,(w,h))

# edge=cv2.canny(image,100,200)
# blur=cv2.GaussianBlur(image,(15,15),0)
# cv2.imshow(image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import cv2 
# import matplotlib.pyplot as plt

# image = cv2.imread("C:/rokey/python/ch21/task/sample2.jpg")
# path = cv2.data.haarcascades + "harrcascade.frontalface_default.xml"
# #face = cv2.casCadeClassifier(path)
# gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# #faces = face.detectMultiScale(gray,scaleFactor=1.0,minNeigbors=4,minSize=(30,30))
# for (x,y,w,h)in faces:
#     #cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
# cv2.imshow(image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import cv2
# video=cv2.VideoCapture(0)
# while True:
#     #ret,frame = video.read()
#     if not ret:
#         break
#     edge=cv2.canny(frame,100,200)
#     cv2.window(edge)
#     if cv2.waitKey(1)==ord('q'):
#         break
# cv2.destroyAllWindows()

# import cv2
# import numpy as np
# image = cv2.imread("C:/rokey/python/ch21/task/sample2.jpg")
# low=np.array([35,100,100])
# upp=np.array([85,255,255])
# green=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
# mask=cv2.inRange(green,low,upp)
# result=cv2.bitwise_and(image,image,mask=mask)
# cv2.imshow('',result)
# cv2.waitKey()
# cv2.destroyAllWindows()