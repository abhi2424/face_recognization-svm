from sklearn import svm,metrics
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mimg
#########################################################training strats##############################################################################################
num_sample=7
train_data=np.zeros((num_sample*40,10304))
train_target=np.zeros((num_sample*40))
c1=-1
for i in range(1,41):
    for j in range(1,num_sample+1):
        c1=c1+1
        im=mimg.imread('C:\\Users\\CG-DTE\\Downloads\\orl_face\\orl_face\\u%d\\%d.png'%(i,j))
        feat=im.reshape(1,-1)
        train_data[c1,:]=feat
        train_target[c1]=i
        #plt.figure(1)
        #plt.imshow(im,cmap='gray')
        #plt.axis('off')
        #plt.title(['user no',str(i),',samp no',str(j)])
        #plt.pause(0.3)
##########################################################training ends#############################################################################################        
##########################################################testing starts##############################################################################################        
test_data=np.zeros(((10-num_sample)*40,10304))
test_target=np.zeros(((10-num_sample)*40))
c1=-1
for i in range(1,41):
    for j in range(num_sample+1,10+1):
        c1=c1+1
        im=mimg.imread('C:\\Users\\CG-DTE\\Downloads\\orl_face\\orl_face\\u%d\\%d.png'%(i,j))
        feat=im.reshape(1,-1)
        test_data[c1,:]=feat
        test_target[c1]=i
svm_model=svm.SVC(kernel='rbf')
svm_model=svm_model.fit(train_data,train_target)
output=svm_model.predict(test_data)
acc=metrics.accuracy_score(test_target,output)
print('accuracy is ::',acc)
print('confusion maetrics')
print(metrics.confusion_matrix(test_target,output))
print('report is')
print(metrics.classification_report(test_target,output))
          
