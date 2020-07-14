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
finalclass = 2;
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
SamplePathes, finalUseLabel = prepData.MakeConv3dSample(RowsCols, allLabel, ImageData, ExtentNum);
#训练集和测试集的划分（分层随机划分，测试集比例0.4）
X_train, X_test, y_trainfinal, y_testfinal = prepData.SplitSample3D(SamplePathes, finalUseLabel, 0.2,
                                                                    finalclass,WS);
if (IsTrain == True):
    # 调用自己搭建的的-CNN模型，用训练集进行训练和拟合
    CNNmodel = ModelCFP.ModelcompileFit(modelCNN.Conv3DCNN(WS, finalclass),
                                        X_train, y_trainfinal, 0.4);
    CNNmodel.save(r'D:\lunwen3Localresult\zongxian\ZXCNN3D22Class3times2band20ext3.h5');
else:
    LoadCNNmodel = load_model(r'D:\lunwen3Localresult\zongxian\ZXCNN3D22Class3times2band20ext3.h5');
    # 利用测试集进行精度评价，并利用模型对所有数据进行分类
    resultLabel, score, lenthTest, Conv2DCNNHHmat = ModelCFP.PredictALLdata3D(ImageData,ExtentNum,X_test, y_testfinal,
                                                                             LoadCNNmodel,Rows, Cols, 1);
    prepData.Draw2orMoreClassResult(resultLabel, 'CNNTEMP');
    print(score)
    print(lenthTest)
    print(Conv2DCNNHHmat)
    sio.savemat(r'D:\lunwen3Localresult\zongxian\ZXCNN3D22Class3times2band20ext3.mat', {'DLMC': resultLabel});
endtime = datetime.datetime.now();
print((endtime - starttime).seconds);