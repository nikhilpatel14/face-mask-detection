import cv2
import numpy as np
withmask=np.load('withmask.npy')
wihtout=np.load('withoutmask.npy')
wihtout=wihtout.reshape(200,100*100*3)
withmask=withmask.reshape(200,100*100*3)
X=np.r_[wihtout,withmask]
print(X.shape)
labels=np.zeros(X.shape[0])
labels[200:]=1.0
name={0:'without',1:'withmask'}
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import  train_test_split
x_train , x_test , y_train ,y_test=train_test_split(X,labels,test_size=0.20)
print(x_train.shape)
from sklearn.decomposition import PCA
pca=PCA(n_components=3)
x_train=pca.fit_transform(x_train)
x_test=pca.transform(x_test)
print(x_train[0])
print(x_train.shape)
svm=SVC()
svm.fit(x_train,y_train)
y_pred=svm.predict(x_test)
print(accuracy_score(y_test,y_pred))
haar_data=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
capture=cv2.VideoCapture(0)
font=cv2.FONT_HERSHEY_COMPLEX
while True:
    flag,img=capture.read()
    if flag:
        face_data = haar_data.detectMultiScale(img)
        for x, y, w, h in face_data:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0),4)
        faces = img[y:y + h, x:x + w, :]
        faces=cv2.resize(faces,(100,100))
        faces = faces.reshape(1, -1)
        faces=pca.transform(faces)
        prediction=svm.predict(faces)
        res=name[int(prediction)]
        print(res)
        if prediction==0:
            cv2.putText(img , res ,(x,y), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,255) , 2)
        else:
            cv2.putText(img, res, (x, y), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
        cv2.imshow('facess', img)
        if cv2.waitKey(2)==27:
            break
capture.release()
cv2.destroyAllWindows()

