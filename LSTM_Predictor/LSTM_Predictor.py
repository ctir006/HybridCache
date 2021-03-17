from numpy import array
from numpy import hstack,vstack
import numpy as np
from keras.models import Sequential,load_model,save_model
from keras.layers import LSTM,Dense,RepeatVector,TimeDistributed
from sklearn.model_selection import train_test_split
from pandas import Series
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from math import sqrt
from keras.constraints import nonneg
import numpy as np
import time

class LSTMPredictor:
    def __init__(self):
        self.predicted=None
        self.actual=None
        self.position=0
		
		
    def split_sequence(self,dataset,n_steps_in,n_steps_out):
        X,y = list(),list()
        for i in range(1,len(dataset)):
            end_ix = i+n_steps_in
            out_end_ix = end_ix+n_steps_out
            if out_end_ix > len(dataset):
                break
            seq_x,seq_y = dataset[i:end_ix,:],dataset[end_ix:out_end_ix,:]
            X.append(seq_x)
            y.append(seq_y)
        return array(X),array(y)

    def train_LSTMModel(self):
        dataset=np.loadtxt("data.txt",dtype=int) 	
        print("Before scaling\n",dataset)
        col_mean = np.nanmean(dataset, axis=0)
        inds = np.where(dataset==0)
        dataset[inds] = np.take(col_mean, inds[1])
		#print(len(dataset),len(dataset[0]))		694 X 1611
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler = scaler.fit(dataset)
        dataset = scaler.transform(dataset)
        print("After Scaling\n",dataset)
        n_steps_in,n_steps_out = 250,20
        X,y = self.split_sequence(dataset,n_steps_in,n_steps_out)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        x1=X_test[self.position].reshape((1,X_test.shape[1],X_test.shape[2]))
        y1=y_test[self.position].reshape((1,y_test.shape[1],y_test.shape[2]))
        n_features = X.shape[2]
        model = Sequential()
        model.add(LSTM(512,activation='relu',input_shape=(n_steps_in,n_features)))
        model.add(RepeatVector(n_steps_out))
        model.add(LSTM(128,activation='relu',return_sequences=True))
        model.add(TimeDistributed(Dense(n_features, activation='linear', W_constraint=nonneg())))
        model.compile(optimizer='sgd',loss='mean_squared_logarithmic_error',metrics=['mse'])
        model.fit(X_train,y_train,epochs=100,batch_size=128,verbose=0)
        model.save_weights("savedModel")
        #model.load_weights("savedModel")
        scores = model.evaluate(X_test, y_test, verbose=0)
        train_mse = model.evaluate(X_train, y_train, verbose=0)
        yhat=model.predict(x1)
        yhat=scaler.inverse_transform(yhat.reshape((y_test.shape[1],y_test.shape[2])))
        y1=scaler.inverse_transform(y1.reshape(y_test.shape[1],y_test.shape[2]))
        self.predicted=yhat
        self.actual=y1
        self.position+=15
        print("Predicted : ",yhat.astype(int))
        print("Actual : ",y1)
        return scores

if __name__=="__main__":
    start_time=time.time()
    obj=LSTMPredictor()
    print(obj.train_LSTMModel())
    print("--- %s seconds ---" % (time.time() - start_time))
    
    
'''
Predicted :  [[ 389   78   68 ...    0    0    0]
 [ 669  132  116 ...    0    0    0]
 [ 878  173  151 ...    0    0    0]
 ...
 [1514  295  254 ...    0    0    0]
 [1517  295  254 ...    0    0    0]
 [1519  296  254 ...    0    0    0]]
Actual :  [[3389. 1661.  505. ...    0.    0.    0.]
 [3729. 1936.  498. ...    0.    0.    0.]
 [2995. 1658.  370. ...    0.    0.    0.]
 ...
 [2080.  709.  383. ...    0.    0.    0.]
 [2283.  781.  361. ...    0.    0.    0.]
 [2813. 1007.  403. ...    0.    0.    0.]]
[0.009869961626827716, 0.013864538993905572]
--- 1383.7867217063904 seconds ---

Replacing zeros with average values in the column

Predicted :  [[ 399  181  125 ...    0    0    0]
 [ 576  283  164 ...    0    0    0]
 [ 703  356  193 ...    0    0    0]
 ...
 [1046  553  278 ...    0    0    0]
 [1047  554  278 ...    0    0    0]
 [1048  554  279 ...    0    0    0]]
Actual :  [[3389. 1661.  505. ...    0.    0.    0.]
 [3729. 1936.  498. ...    0.    0.    0.]
 [2995. 1658.  370. ...    0.    0.    0.]
 ...
 [2080.  709.  383. ...    0.    0.    0.]
 [2283.  781.  361. ...    0.    0.    0.]
 [2813. 1007.  403. ...    0.    0.    0.]]
[0.009217792497399976, 0.012939486312953864]
--- 1413.7850341796875 seconds ---

'''
		



