# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 21:19:16 2016
@author: Danni Cheng
Source: https://github.com/Tristacheng/mutimodality_cnn/blob/master/base_cnn.py
"""

#from __future__ import print_function
import os
import timeit
#import sys
import numpy as np
#import gzip
np.random.seed(1234)  # for reproducibility

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense,Dropout, Activation, Flatten
from keras.layers import Convolution3D, MaxPooling3D
#from keras.layers.noise import GaussianDropout
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from keras.regularizers import WeightRegularizer, l1, l2, l1l2
#from keras.callbacks import Callback
from keras.optimizers import Adadelta
#from keras.optimizers import SGD
from keras.utils import np_utils
#from keras.utils.visualize_util import plot
from sklearn.svm import SVC
import six.moves.cPickle as pickle
from sklearn import grid_search
from sklearn import metrics

import scipy.io as sio  


#def load_data(data_i):
#    data_i = data_i+1
#    dataSetpath = '/media/cdn/E/cdn/DataSet'
##    numd = 4
##    if (data_i<=numd):
##        filename = 'Smooth64\\AffineADNC_64s_'+str(data_i)+'.mat'
##        full_filename= os.path.join(dataSetpath,filename)
##        dataMat = sio.loadmat(full_filename)
##        AD_name = 'AffineAD_64s_'+str(data_i)
##        AD_data = dataMat[AD_name]
##        num_AD =  AD_data.shape[0]
##        NC_name = 'AffineNC_64s_'+str(data_i)
##        NC_data = dataMat[NC_name]
##        num_NC =  NC_data.shape[0]
##    else:
##        data_i = data_i - numd
#    filename = 'MRIandPET/AffineADNC_64MRI_'+str(data_i)+'.mat'
#    full_filename= os.path.join(dataSetpath,filename)
#    dataMat = sio.loadmat(full_filename)
#    AD_name = 'AffineAD_MRI'
#    AD_data = dataMat[AD_name]
#    num_AD =  AD_data.shape[0]
#    NC_name = 'AffineNC_MRI'
#    NC_data = dataMat[NC_name]
#    num_NC =  NC_data.shape[0]
#    X_size,Y_size,Z_size = NC_data.shape[1:]
#    del dataMat
#    numAllSample = num_AD+num_NC
#    adLabel =  np.ones((num_AD,))
#    normalLabel =  np.zeros((num_NC,))
#    imgADNC_3D = np.zeros((numAllSample,49 , 40 , 39))
#    imgADNC_3D[:num_AD,:X_size,:Y_size,:Z_size] = AD_data
#    imgADNC_3D[num_AD:,:X_size,:Y_size,:Z_size] = NC_data
#    imgVector = imgADNC_3D.reshape(numAllSample,-1)
#    imgVector = imgVector.astype('float32')
#    imgVector = (imgVector-np.min(imgVector))*2.0/(float(np.max(imgVector)-np.min(imgVector)))-1
#    data3dvt = imgVector
#    imgLabel = np.hstack((adLabel,normalLabel))
#    return data3dvt,imgLabel
    
    
def load_data(patch_i,data_i):
    data_i = data_i+1
    dataSetpath = '/media/disk_e/PET_PatchDATA'
    filename = 'AffineADNC64_Patch'+str(patch_i)+'_PET_'+str(data_i)+'.mat'
    full_filename= os.path.join(dataSetpath,filename)
    dataMat = sio.loadmat(full_filename)
    AD_name = 'PatchAD'
    AD_data = dataMat[AD_name]
    num_AD =  AD_data.shape[0]
#    print num_AD
    NC_name = 'PatchNC'
    NC_data = dataMat[NC_name]
    num_NC =  NC_data.shape[0]
#    print num_NC
    X_size,Y_size,Z_size = NC_data.shape[1:]
    del dataMat
    numAllSample = num_AD+num_NC
    adLabel =  np.ones((num_AD,))
    normalLabel =  np.zeros((num_NC,))
    imgADNC_3D = np.zeros((numAllSample,49,39,38))
    imgADNC_3D[:num_AD,:X_size,:Y_size,:Z_size] = AD_data[:,:49,:39,:38]
    imgADNC_3D[num_AD:,:X_size,:Y_size,:Z_size] = NC_data[:,:49,:39,:38]
    imgVector = imgADNC_3D.reshape(numAllSample,-1)
    imgVector = imgVector.astype('float16')
    imgVector = (imgVector-np.min(imgVector))*2.0/(float(np.max(imgVector)-np.min(imgVector)))-1
    data3dvt = imgVector
    imgLabel = np.hstack((adLabel,normalLabel))
#    print len(imgLabel)
    return data3dvt,imgLabel    

def load_cv(cv_name = 'index_10fold.pkl',fold_idx = 0):
    input_doc = open(cv_name,'rb')
    in_data = pickle.load(input_doc)
    input_doc.close()
    train_idx,test_idx = in_data
    train_index0 = train_idx[fold_idx]
    test_index = test_idx[fold_idx]
    val_index = test_idx[fold_idx-1]
    train_index = list(set(train_index0)-set(val_index))
    train_index = np.array(train_index)
    return train_index,test_index,val_index 

def svc(traindata,trainlabel,testdata,testlabel):
    print("Start training SVM...")
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    clf = grid_search.GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    clf = clf.fit(traindata,trainlabel)
    print("Best estimator found by grid search:")
    print(clf.best_estimator_)
    pred_testlabel = clf.predict(testdata)
    return pred_testlabel
    
def data_train0_test1_val2(aug_time,train0test1val2,fold_idx,image_idx,patch_idx):
    train_id,test_id,val_id = load_cv(fold_idx = fold_idx)

    if (train0test1val2 == 0):
        return_index = train_id
    elif (train0test1val2 == 1):
        return_index = test_id
    else :
        return_index = val_id
    if (aug_time==1):
        dataVct,image_label = load_data(patch_idx,image_idx) 
        dataVct = dataVct[return_index]
        data_label = image_label[return_index]
    else:
        for i in range(aug_time):
            if (i==0):
                dataVct,image_label = load_data(patch_idx,i) 
                dataVct = dataVct[return_index]
                data_label = image_label[return_index]
            else:
                Vct,image_label = load_data(patch_idx,i)
                Vct = Vct[return_index]
                dataVct = np.concatenate((dataVct,Vct),axis = 0)
                img_label = image_label[return_index]
                data_label = np.hstack((data_label,img_label))
#    print len(data_label)
#    if (train0test1val2 == 0)and(aug_time!=1):
#        for i in range(aug_time):
#            if (i==0):
#                dataVct_MRIonly,data_label_MRIonly = load_MRIonly(i) 
#            else:
#                Vct_MRIonly,image_label = load_MRIonly(i)
#                dataVct_MRIonly = np.concatenate((dataVct_MRIonly,Vct_MRIonly),axis = 0)
#                data_label_MRIonly = np.hstack((data_label_MRIonly,image_label))
#        data_label = np.hstack((data_label,data_label_MRIonly)) 
#        dataVct = np.concatenate((dataVct,dataVct_MRIonly),axis = 0)
#    else:
#        data_label = data_label
#        dataVct = dataVct
    np.random.seed(1234)   
    random_idx = np.random.permutation(range(len(data_label)))            
    dataX = dataVct[random_idx]
    dataY = data_label[random_idx]
    length_v = data_label.shape[0]        
    dataX = dataX.reshape(length_v, 1,49,39,38)
    if train0test1val2==1:
        randomIndex = return_index[random_idx]
    else:
        randomIndex = random_idx

    return dataX,dataY,randomIndex
    

def create_model(model_para = [6, 12, 18, 18, 64, 0.01, 0.4,(5, 5, 5)],foldname = os.path.abspath('.') ):
    # input image dimensions
#    img_rows, img_cols, img_depth = 38, 39, 48
    img_rows, img_cols, img_depth = 49,39,38
    # number of convolutional filters to use
    conv_l2 = 0.008
    
    full_l2  = 0.3
    # convolution kernel size
    kernel_size = (3,3,3)
    # size of pooling area for max pooling
    pool_size = (2, 2, 2)

    drop_out = (model_para[5], model_para[6])
    
    act_function = 'tanh'
    
    full_connect = model_para[4]
    
    nb_filters = (model_para[0], model_para[1], model_para[2], model_para[3])
#    nb_filters = (5, 10, 15, 15)
    l1_regularizer = 0.01
    
    l2_regularizer = full_l2
    
    nb_classes = 2
    
    input_shape = (1, img_rows, img_cols, img_depth)
    
    wr = WeightRegularizer(l1=l1_regularizer,l2=l2_regularizer)
    #creat cnn model
    model = Sequential()
    
    model.add(Convolution3D(nb_filters[0], kernel_size[0], kernel_size[1],kernel_size[2],W_regularizer = l2(conv_l2),
                            activation = act_function , input_shape=input_shape))                  
    model.add(MaxPooling3D(pool_size=pool_size)) 

    model.add(Dropout(drop_out[0]))
    
    model.add(Convolution3D(nb_filters[1], kernel_size[0], kernel_size[1], kernel_size[2],W_regularizer = l2(conv_l2),
                               activation = act_function))
    model.add(MaxPooling3D(pool_size=pool_size))
    
    model.add(Dropout(drop_out[0]))
    
    model.add(Convolution3D(nb_filters[2], kernel_size[0], kernel_size[1], kernel_size[2],W_regularizer = l2(conv_l2),
                                activation = act_function))                    
    model.add(MaxPooling3D(pool_size=pool_size))    
    
    model.add(Dropout(drop_out[0]))

    model.add(Convolution3D(nb_filters[3], kernel_size[0], kernel_size[1], kernel_size[2],W_regularizer = l2(conv_l2),
                            activation = act_function))
#    model.add(MaxPooling3D(pool_size=pool_size))    
    
    model.add(Dropout(drop_out[1]/2))
    
    model.add(Flatten())
    
    model.add(Dense(full_connect, W_regularizer = wr,activation = act_function))

    model.add(Dropout(drop_out[1]))
    
    model.add(Dense(nb_classes,activation = act_function))
    
    model.add(Activation('softmax'))
#    model.add(Activation(act_function))
    model.summary()
    
    ADA = Adadelta(lr = 2.0, rho=0.95)
    
    model.compile(loss= 'categorical_crossentropy',
              optimizer= ADA,
              metrics=['accuracy'])
              
##    save parameters of cnn model to .txt             
    sname = 'model_parameter.txt'
    full_namem = os.path.join(foldname,sname)
    fm = open(full_namem,'wb')
    fm.write('************CNN model parameter************ '+'\n')
    fm.write('Number of Convolution layer :     '+str(len(nb_filters))+'\n')
    fm.write('Input shape :                     '+str(input_shape)+'\n')
    fm.write('Number of kernal per layer ï¼?    '+str(nb_filters)+'\n')
    fm.write('Kernel size per layer :           '+str(kernel_size)+'\n')
    fm.write('Pool size per layer :             '+str(pool_size)+'\n')
    fm.write('Activation function per layer :   '+act_function+'\n')
#    fm.write('Dropout rate :                    '+str(drop_out)+'\n')
    fm.write('Number of full-connect layer :    '+str(full_connect)+'\n')
    fm.write('Coefficient of L1 regularizer :   '+str(l1_regularizer)+'\n')
    fm.write('Coefficient of L2 regularizer :   '+str(l2_regularizer)+'\n')
    fm.write('Output :                          '+str(nb_classes)+' classes'+'\n')
    fm.close()
    
    return model
    

start_time = timeit.default_timer()    
nb_classes = 2
n_fold = 10
curpath = os.path.abspath('.')

img_rows, img_cols, img_depth = 49,39,38  

nb_epoch = 80
bsize = 64           
augtime = 8
test_score = ['']*augtime
Ytest_prd = ['']*augtime
#param = [[15,25,50,50,35, 0.10, 0.6,(5, 5, 5)],
#          [15,25,60,60,40, 0.10, 0.5,(5, 5, 5)],
#          [15,25,60,60,40, 0.15, 0.5,(5, 5, 5)],
#           [15,25,40,40,30, 0.10, 0.6,(5, 5, 5)],
#           [15,25,40,40,30, 0.10, 0.5,(5, 5, 5)],
#          [15,25,60,60,50, 0.10, 0.6,(5, 5, 5)],
#           [15,25,60,60,30, 0.10, 0.5,(5, 5, 5)],
#            [15,25,40,60,40, 0.10, 0.6,(5, 5, 5)],
#                [15,25,40,50,40, 0.10, 0.5,(5, 5, 5)]]
#testpara = len(param)
testpara = 27
record_train_acc = np.zeros((testpara,n_fold))
record_test_acc = np.zeros((testpara,n_fold))
record_val_acc = np.zeros((testpara,n_fold))
record_train_loss = np.zeros((testpara,n_fold))
record_test_loss = np.zeros((testpara,n_fold))
record_val_loss = np.zeros((testpara,n_fold))
record_svm_acc = np.zeros((testpara,n_fold))
#for pri in range(testpara):
#n = [25,26]
for rii in range(27):
    t1_time = timeit.default_timer()
    filtersize = (3,3,3)
    pri = rii
    numi = 0
    saveSC = np.zeros((n_fold,2))

    saveYP=['']*n_fold
    saveYP_svm =['']*n_fold
    saveSC_svm =np.zeros((n_fold,1))
    title = os.path.split(__file__)[1][:-3] 
    sub_title = 'Patch_'+str(rii)
    new_fold0 = os.path.join(curpath,title)
    new_fold = os.path.join(new_fold0,sub_title)
    if not os.path.isdir(new_fold):
        os.makedirs(new_fold)
    name_para = 'train_para_'+str(rii)+'.txt'
    full_namep = os.path.join(new_fold,name_para)
    fp = open(full_namep,'wb')
    fp.write('************  CNN training parameter  ************ '+'\n')
    #    fp.write('Data name:           '+str(filename)+'\n')
    #    fp.write('3D image number:     '+str(dataX.shape[0])+'\n')
    #    fp.write('Image name:          '+str(dataX.shape[1:])+'\n'+'\n')
    fp.write('Fold number:         '+str(n_fold)+'\n')
    fp.write('Number of epoch:     '+str(nb_epoch)+'\n')
    fp.write('Batch size:          '+str(bsize)+'\n')
    fp.write('Training Targe:      '+str(nb_classes)+' classes'+'\n')
    fp.close()
    
    for fold_i in range(n_fold):
        np.random.seed(1234)
        print('========== Now running on fold '+ str(fold_i+1) + ' ==========')
        X_train,Y_train,rt = data_train0_test1_val2(aug_time = augtime,train0test1val2 = 0,fold_idx = fold_i,image_idx = 0,patch_idx = rii+1)
        X_test,Y_test,rdxIdx = data_train0_test1_val2(aug_time = 1,train0test1val2 = 1,fold_idx = fold_i,image_idx = 0,patch_idx = rii+1)
        X_val,Y_val,rv = data_train0_test1_val2(aug_time = augtime,train0test1val2 = 2,fold_idx = fold_i,image_idx = 0,patch_idx = rii+1)
        
        Y_train = np.array(Y_train, dtype='float32')
        Y_val = np.array(Y_val, dtype='float32')
        Y_train_n = np_utils.to_categorical(Y_train, nb_classes)
        Y_test_n = np_utils.to_categorical(Y_test, nb_classes)
        Y_val_n = np_utils.to_categorical(Y_val, nb_classes)
        
        model = create_model(model_para = [15,25,50,50,40, 0.10, 0.6,(3, 3, 3)], foldname = new_fold)
    
        name_wts = 'weights_'+str(fold_i+1)+'.hdf5'
        name_log = 'training_'+str(fold_i+1)+'.log'
        traininglog = os.path.join(new_fold,name_log)
        best_weight_name = os.path.join(new_fold,name_wts)
        csv_logger = CSVLogger(traininglog)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                  patience=3, min_lr=0.0001)
        checkpointer = ModelCheckpoint(filepath=best_weight_name, verbose=1, save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)

        logif = model.fit(X_train, Y_train_n, batch_size=bsize, nb_epoch=nb_epoch, shuffle = False,
           verbose=1,validation_data=(X_val,Y_val_n),callbacks = [csv_logger,reduce_lr,checkpointer,early_stopping])
        del X_train
        
        loss_history = logif.history
        name_log = 'loss_history_'+str(fold_i+1)+'.txt'
        full_log_name = os.path.join(new_fold,name_log)
        logf = open(full_log_name,'w')
        
        for keyi in loss_history:
            write_str = keyi + ': ' + str(loss_history[keyi]) + '\n'
            logf.write(write_str)  
        logf.close()
        
        model.load_weights(best_weight_name)
        test_score = model.evaluate(X_test, Y_test_n,batch_size=bsize, verbose=1)
        Ytest_prd = model.predict(X_test,batch_size=bsize)
        print('Test score' +':', test_score[0])
        print('Test accuracy' +':', test_score[1])
        
#        record_test_acc_all[pri,fold_i] = test_score[1]
#        record_test_loss_all[pri,fold_i] = test_score[0]
        X_val,Y_val,rv = data_train0_test1_val2(aug_time = 1,train0test1val2 = 2,fold_idx = fold_i,image_idx = 0,patch_idx = rii+1)
        Y_val_n = np_utils.to_categorical(Y_val, nb_classes)
	val_score = model.evaluate(X_val, Y_val_n,batch_size=bsize, verbose=1)
        val_prd = model.predict(X_val,batch_size=bsize)

        X_train,Y_train,train_rdnidx = data_train0_test1_val2(aug_time = 1,train0test1val2 = 0,fold_idx = fold_i,image_idx = 0,patch_idx = rii+1)
        Y_train = np.array(Y_train, dtype='float32')
        Y_train_n = np_utils.to_categorical(Y_train, nb_classes)
        train_score = model.evaluate(X_train, Y_train_n,batch_size=bsize, verbose=1)
        Ytrain_prd = model.predict(X_train,batch_size=bsize)
        
        print('Train score:', train_score[0])
        print('Train accuracy:', train_score[1])
        record_train_acc[pri,fold_i] = train_score[1]
        record_train_loss[pri,fold_i] = train_score[0]
        
        print('Val score:', val_score[0])
        print('Val accuracy:', val_score[1])
        record_val_acc[pri,fold_i] = val_score[1]
        record_val_loss[pri,fold_i] = val_score[0]
        
        print('Test score:', test_score[0])
        print('Test accuracy:', test_score[1])
        record_test_acc[pri,fold_i] = test_score[1]
        record_test_loss[pri,fold_i] = test_score[0]
        saveSC[fold_i,:]=test_score[0] 
        saveYP[fold_i]=Ytest_prd[0]

        json_string = model.to_json()
        name_json = 'model_architecture_'+str(fold_i+1)+'.json'
        full_namej = os.path.join(new_fold,name_json)
        open(full_namej,'w').write(json_string)  
        name_model = 'model_weights_'+str(fold_i+1)+'.h5'
        full_namem = os.path.join(new_fold,name_model)
        model.save_weights(full_namem)
##        define theano funtion to get output of FC layer
        get_feature = K.function([model.layers[0].input,K.learning_phase()],[model.layers[12].output])
        get_feature1 = K.function([model.layers[0].input,K.learning_phase()],[model.layers[9].output])
        get_feature2 = K.function([model.layers[0].input,K.learning_phase()],[model.layers[7].output])      
#        get_feature3 = K.function([model.layers[0].input,K.learning_phase()],[model.layers[5].output])
        FC_train_feature = np.zeros(())
        FC_train_feature1 = np.zeros(())
        FC_train_feature2 = np.zeros(())
#        FC_train_feature3 = np.zeros(())
        train_num = X_train.shape[0]
        X_train_d = ['']*3
        Y_train_d = ['']*3
        X_train_d[0] = X_train[:train_num//3]
        Y_train_d[0] = Y_train[:train_num//3]
        X_train_d[1] = X_train[train_num//3:train_num//3*2]
        Y_train_d[1] = Y_train[train_num//3:train_num//3*2]
        X_train_d[2] = X_train[train_num//3*2:]
        Y_train_d[2] = Y_train[train_num//3*2:]
        for i in range(3):
            if (i == 0):
                FC_train_feature = get_feature([X_train_d[0],1])[0]
                FC_train_feature1 = get_feature1([X_train_d[0],1])[0]
                FC_train_feature2 = get_feature2([X_train_d[0],1])[0]
#                FC_train_feature3 = get_feature3([X_train_d[0],1])[0]
                Y_train_new = Y_train_d[0]
            else:
                FC_train_feature = np.concatenate((FC_train_feature,get_feature([X_train_d[i],1])[0]),axis = 0) 
                Y_train_new = np.concatenate((Y_train_new,Y_train_d[i]),axis = 0)
                FC_train_feature1 = np.concatenate((FC_train_feature1,get_feature1([X_train_d[i],1])[0]),axis = 0)
                FC_train_feature2 = np.concatenate((FC_train_feature2,get_feature2([X_train_d[i],1])[0]),axis = 0)
#                FC_train_feature3 = np.concatenate((FC_train_feature3,get_feature3([X_train_d[i],1])[0]),axis = 0)
        FC_test_feature = get_feature([X_test,0])[0]
        FC_test_feature1 = get_feature1([X_test,0])[0]
        FC_test_feature2 = get_feature2([X_test,0])[0]
#        FC_test_feature3 = get_feature3([X_test,0])[0]
        Y_test_new  = Y_test
        FC_val_feature = get_feature([X_val,0])[0]
        FC_val_feature1 = get_feature1([X_val,0])[0]
        FC_val_feature2 = get_feature2([X_val,0])[0]
#        FC_val_feature3 = get_feature3([X_val,0])[0]
        Y_val_new  = Y_val
        name_feat = 'model_fold_'+str(fold_i+1)+'.mat'
        name_feat1 = 'model_flatten_l1_'+str(fold_i+1)+'.mat'
        name_feat2 = 'model_flatten_l2_'+str(fold_i+1)+'.mat'  
#        name_feat3 = 'model_flatten_l3_'+str(fold_i+1)+'.mat'
        full_feat_mat = os.path.join(new_fold,name_feat)
        full_feat_mat1 = os.path.join(new_fold,name_feat1)
        full_feat_mat2 = os.path.join(new_fold,name_feat2)        
#        full_feat_mat3 = os.path.join(new_fold,name_feat3)       
        sio.savemat(full_feat_mat,{'train_feature':FC_train_feature,'train_label':Y_train_new,
                                   'val_feature':FC_val_feature,'val_label':Y_val_new,
                                   'test_feature':FC_test_feature,'test_label':Y_test_new})
        sio.savemat(full_feat_mat1,{'train_feature':FC_train_feature1,'train_label':Y_train_new,
                                   'val_feature':FC_val_feature1,'val_label':Y_val_new,
                                   'test_feature':FC_test_feature1,'test_label':Y_test_new})
        sio.savemat(full_feat_mat2,{'train_feature':FC_train_feature2,'train_label':Y_train_new,
                                   'val_feature':FC_val_feature2,'val_label':Y_val_new,
                                   'test_feature':FC_test_feature2,'test_label':Y_test_new})
#        sio.savemat(full_feat_mat3,{'train_feature':FC_train_feature3,'train_label':Y_train_new,
#                                   'val_feature':FC_val_feature3,'val_label':Y_val_new,
#                                   'test_feature':FC_test_feature3,'test_label':Y_test_new})
        
        pred_label = svc(FC_train_feature,Y_train_new,FC_test_feature,Y_test_new)
        svm_acc = metrics.accuracy_score(pred_label,Y_test_new)
        svm_recall = metrics.recall_score(pred_label,Y_test_new)
        svm_precision = metrics.precision_score(pred_label,Y_test_new)
        print('CNN - SVM accuracy score:', svm_acc)
        print('CNN - SVM recall score:', svm_recall)
        print('CNN - SVM precision score:', svm_precision)
#        saveSC_svm[fold_i,:]=svm_acc 
        record_svm_acc[pri,fold_i] = svm_acc 
        saveYP_svm[fold_i]=pred_label
        if fold_i==0:
            test_pro = Ytest_prd[:,1]
#            test_pro = test_p.reshape((test_p.shape[0],1))
            rd_test = rdxIdx
            true_test = Y_test
        else:
            test_p = Ytest_prd[:,1]
#            test_p = test_p.reshape((test_p.shape[0],1))
            test_pro = np.concatenate((test_pro,test_p),axis =0)
            rd_test = np.concatenate((rd_test,rdxIdx),axis=0)
            true_test = np.concatenate((true_test,Y_test),axis=0)
#        numi = numi+1
#    name_sv = 'result'+'.pkl'
    name_sv = 'result'+'.mat'
    full_namesv = os.path.join(new_fold,name_sv)
    sio.savemat(full_namesv,{'test_prob':test_pro,'random_test_index':rd_test,'true_label':true_test})
#    fr = open(full_namesv,'wb')        
#    outs = [saveYP,saveSC,saveSC_svm,saveYP_svm]
#    pickle.dump(outs,fr)
#    fr.close()
    name_rs = 'result_'+'run_para' +str(pri)+'.txt'
    full_namers = os.path.join(new_fold,name_rs)
    ft = open(full_namers,'wb')
    ft.write('*************  Result of CNN model  ************ '+'\n'+'\n')
    ft.write('Loss and accuracy :'+ '\n')
    for idl in range(n_fold):
        ft.write('Fold  '+str(idl+1) +'\n')
        ft.write('   Test  :'+ str(record_test_acc[pri,idl]) +'   '+ str(record_test_loss[pri,idl]) +'\n')
        ft.write('   Train :'+ str(record_train_acc[pri,idl]) +'   '+ str(record_train_loss[pri,idl]) +'\n')
        ft.write('   Val   :'+ str(record_val_acc[pri,idl]) +'   '+ str(record_val_loss[pri,idl]) +'\n')
    dacc = np.mean(record_test_acc[pri,:])
    avg_loss = np.mean(record_test_loss[pri,:])
    ft.write('Test accuracy for whole data:   %f %%' %(dacc*100.)+'\n')
    ft.write('The average loss:    '+ str(avg_loss)+'\n'+'\n')
    
    ft.write('************  Test result of SVM from CNN feature  ************ '+'\n'+'\n')
    ft.write('Test accuracy :'+ '\n')
    for idl in range(n_fold):
        ft.write('Fold '+str(idl+1)+ ':   '+str(record_svm_acc[pri,idl]) +'\n')
    dacc_svm = np.mean(record_svm_acc[pri,:])
    ft.write('Test accuracy for whole data:   %f %%' %(dacc_svm*100.)+'\n')
    t2_time = timeit.default_timer()
    ft.write(('The code for this parameter ran for %.2fm' % ((t2_time -t1_time) / 60.)))
    ft.close()

    end_time = timeit.default_timer()
    finalresult_name = os.path.join(new_fold0,'final_result.mat')
#    finalresult_name_aug = os.path.join(new_fold0,'final_result_for_aug.mat')
    avg_train_acc = np.mean(record_train_acc,axis = 1)
    avg_train_acc = avg_train_acc.T
    avg_test_acc = np.mean(record_test_acc,axis = 1)
    avg_test_acc = avg_test_acc.T
    avg_val_acc = np.mean(record_val_acc,axis = 1)
    avg_val_acc = avg_val_acc.T
    avg_train_loss = np.mean(record_train_loss,axis = 1)
    avg_test_loss = np.mean(record_test_loss,axis = 1)
    avg_val_loss = np.mean(record_val_loss,axis = 1)
    sio.savemat(finalresult_name,{'train_acc':record_train_acc,'test_acc':record_test_acc,'val_acc':record_val_acc,
                                'train_loss':record_train_loss,'test_loss':record_test_loss,'val_loss':record_val_loss,
                                'avg_train_acc':avg_train_acc,'avg_test_acc':avg_test_acc,'avg_val_acc':avg_val_acc,
                                'avg_train_loss':avg_train_loss,'avg_test_loss':avg_test_loss,'avg_val_loss':avg_val_loss})
#    sio.savemat(finalresult_name_aug,{'test_acc_all':record_test_acc_all,'test_loss_all':record_test_loss_all})
    
    print(('The code for file ' +
    os.path.split(__file__)[1] +
        ' ran for %.2fm' % ((end_time - start_time) / 60.)))
        
sio.savemat('final_result.mat',{'train_acc':record_train_acc,'test_acc':record_test_acc,'val_acc':record_val_acc,
                                'train_loss':record_train_loss,'test_loss':record_test_loss,'val_loss':record_val_loss,
                                'avg_train_acc':avg_train_acc,'avg_test_acc':avg_test_acc,'avg_val_acc':avg_val_acc,
                                'avg_train_loss':avg_train_loss,'avg_test_loss':avg_test_loss,'avg_val_loss':avg_val_loss})
#sio.savemat('final_result_for_aug.mat',{'test_acc_all':record_test_acc_all,'test_loss_all':record_test_loss_all})
