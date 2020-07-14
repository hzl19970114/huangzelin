# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 14:52:27 2018
@author: zhang
"""
from keras.models import load_model
import scipy.io as sio
import allImageDataPath as ALLPATH
import ALLCNNModelFunction as modelCNN
import ReadImageFromPath as prepData
import ModelCompileFitPredict as ModelCFP
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import datetime
finalclass = 8;
ExtentNum = 3;
# 原始影像的路径，读取到8个波段的数据，上采样至15米
# 导入影像数据集和样本数据集
SpatialRef = ALLPATH.SpatialRef                             #导入tempMbands作为空间参考
ImageData = io.imread(ALLPATH.Mutilbands)
shapePath = ALLPATH.SamplePoint;                            #样本点
Rows = ImageData.shape[0];                                  #ImageData行数量
Cols = ImageData.shape[1];                                  #ImageData列数量
numberBands = ImageData.shape[2];                           #ImageData通道数
starttime = datetime.datetime.now();                        #Time库
IsTrain = False;                                             #是否用训练模型
WS = 2 * ExtentNum + 1;
scale = 1
# 获取样本点的行列号以及Label
RowsCols, allLabel = prepData.GetXYValueToRowsCols(SpatialRef, shapePath);
SamplePathes, finalUseLabel = prepData.MakeSample(RowsCols, allLabel, ImageData, ExtentNum);
#训练集和测试集的划分（分层随机划分，测试集比例0.4）
X_train, X_test, y_trainfinal, y_testfinal = prepData.SplitSample(SamplePathes, finalUseLabel, 0.2, finalclass);
if (IsTrain == True):
    # 调用自己搭建的的-CNN模型，用训练集进行训练和拟合
    CNNmodel = ModelCFP.ModelcompileFit(modelCNN.Conv2DCNN(WS, numberBands, finalclass),
                                        X_train, y_trainfinal, 0.4);
    CNNmodel.save(r'E:\last\model\dl\CNN2D2Class3times2band20ext3.h5');
else:
    LoadCNNmodel = load_model(r'E:\last\model\dl\CNN2D2Class3times2band20ext3.h5');
    # 利用测试集进行精度评价，并利用模型对所有数据进行分类
    resultLabel, score, lenthTest, Conv2DCNNHHmat = ModelCFP.PredictALLdata(ImageData,ExtentNum,X_test, y_testfinal,
                                                                             LoadCNNmodel,Rows, Cols, 1);
    prepData.Draw2orMoreClassResult(resultLabel, 'CNNTEMP');
    print(score)
    print(lenthTest)
    print(Conv2DCNNHHmat)
    sio.savemat(r'E:\last\model\dl\fjCNN2D2Class3times2band20ext3.mat', {'DLMC': resultLabel});
endtime = datetime.datetime.now();
print((endtime - starttime).seconds);