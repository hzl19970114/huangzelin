from keras.models import load_model
import scipy.io as sio
import allImageDataPath as ALLPATH
import ALLCNNModelFunction as modelCNN
import ReadImageFromPath as prepData
import ModelCompileFitPredict as ModelCFP
from skimage import io
import numpy as np
finalclass = 2;
IsTrain = False;
SpatialRef = ALLPATH.SpatialRef                             #导入tempMbands作为空间参考
ImageData = io.imread(ALLPATH.Mutilbands);                  #多光谱skimage成多维数组
shapePath = ALLPATH.SamplePoint;                            #样本点
Rows = ImageData.shape[0];                                  #ImageData行数量
Cols = ImageData.shape[1];                                  #ImageData列数量
numberBands = ImageData.shape[2];                           #ImageData通道数
# 获取样本点的行列号以及Label
RowsCols,allLabel = prepData.GetXYValueToRowsCols(SpatialRef,shapePath);
SamplePathes,finalUseLabel = prepData.MakeMLPSample(ImageData, RowsCols, allLabel);
# 训练集和测试集的划分（分层随机划分，测试集比例0.4）
X_train,X_test,y_trainfinal,y_testfinal = prepData.MLPSplitSample(SamplePathes, finalUseLabel, 0.2);
if (IsTrain == False):
    # 调用自己搭建的的-CNN模型，用训练集进行训练和拟合
    CNNmodel = ModelCFP.ModelcompileFit(modelCNN.MLP(numberBands),
                                              X_train, y_trainfinal,0.4);
    CNNmodel.save(r'D:\lunwen2bannanover\DLresult\MLP7Class0723band10.h5');
else:
    LoadCNNmodel = load_model(r'D:\lunwen2bannanover\DLresult\MLP7Class0723band10.h5');
    # 利用测试集进行精度评价，并利用模型对所有数据进行分类
    PredictLabel,score,lenthTest,Conv1DCNNHHmat = ModelCFP.PredictLabelConv1D( ImageData,
                                                     X_test,y_testfinal,LoadCNNmodel,Rows,Cols);
    prepData.Draw2orMoreClassResult(PredictLabel,'MLP');
    print(score)
    print(lenthTest)
    print(Conv1DCNNHHmat)
    sio.savemat(r'D:\lunwen2bannanover\DLresult\MLP7Class0723band10.mat',{'DLMC':PredictLabel});