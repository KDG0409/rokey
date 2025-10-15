#seaborn
import seaborn as sns
import matplotlib.pyplot as plt

iris=sns.load_dataset("iris")
sns.set_theme(style='',palette='')
sns.scatterplot(data=iris,x='',y='',hue='',style='')
sns.set_palette('')
sns.lmplot(data=iris,x='',y='',hue='',height=4)
sns.histplot(data=iris,x='',hue='',kde=True)
sns.boxplot(data=iris,x='',y='',)
sns.violinplot(data=iris,x='',y='',inner='quart')
sns.pairplot(iris,hue='')
g=sns.FacetGrid(iris,col='',row='',height=4,aspect=1.0)
g.map_dataframe(sns.histplot,hue='',kde=True)
g.set_titles(col_templete='',row_templete='')
plt.show()

#opencv
# import cv2
# image=cv2.imread('')
# resized=cv2.resize(image,(500,500))
# gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# (h,w)=image.shape[:2]
# center=(w//2,h//2)
# m=cv2.getRotaionMatrix2D(center,45,1.0)
# rotated=cv2.warpAffine(image,m,(w,h))
# edge=cv2.canny(image,100,200)
# blur=cv2.GaussianBlur(image,(15,15),0)
# cv2.imshow('title',image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#객체 인식
# import cv2
# import matplotlib.pyplot as plt
# image=cv2.imread('')
# path=cv2.data.harrcascades + "haarcascade_frontalface_default.xml"
# face=cv2.CascadeClassifier(path)
# gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# faces=face.detectMultiScale(gray,scaleFactor=1.0,minNeigbor=4,minsize=(50,50))
# for (x,y,w,h) in faces:
#     cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
# cv2.imshow('title',image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#영상 인식
# import cv2
# image=cv2.VideoCapture('')
# while True:
#     ret,frame = image.read()
#     if ret == False:
#         break
#     cv2.window('title',frame)
#     if cv2.waitKey(1) == ord('q'):
#         break
# cv2.destroyAllWindows()

#녹색필터
# import cv2
# import numpy as np
# image=cv2.imread('')
# low=np.array([])
# upp=np.array([])
# green=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
# mask=cv2.inRange(green,low,upp)
# filler=cv2.bitWise_and(image,image,mask=mask)
# cv2.imshow('title',filler)
# cv2.waitKey(0)
# cv2.destroyAllWindows()