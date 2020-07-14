try:
    from osgeo import gdal
    from osgeo import ogr
    from osgeo import osr
except ImportError:
    import gdal
    import ogr
    import osr
# from ospybook as pb
# import scipy.io as sio
import scipy.ndimage as SNDIG
import keras,math
import numpy as np
from skimage import io   #它将图片作为numpy数组进行处理
#from sklearn.feature_extraction import image
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#对于单波段，外围扩充一圈
def ExtentOneCircle(A):
    Rows=A.shape[0];         # Cols=A.shape[1];
    LRE= np.concatenate((A[:,0].reshape(Rows,1),A, A[:,-1].reshape(Rows,1)),axis=1);#左右填充，但好像填充的不是数值0啊？有几种填充方式吗？
    COLSLRE=LRE.shape[1];    # ROWSLRE=LRE.shape[0];
    UpDownExtent=np.concatenate((LRE[0,:].reshape(1,COLSLRE),LRE,LRE[-1,:].reshape(1,COLSLRE)),axis=0);#上下填充
    return UpDownExtent;

#对于单波段，外围扩充1，2，3，4, 5, 6圈
def WindowSizeFunction(A,extentnumber):
    if(extentnumber>=7):
        print("外围扩展次数必须小于7");
    elif(extentnumber==1):
        finalNeibor=ExtentOneCircle(A);
    elif(extentnumber==2):
        finalNeibor=ExtentOneCircle(ExtentOneCircle(A));
    elif(extentnumber==3):
        finalNeibor=ExtentOneCircle(ExtentOneCircle(ExtentOneCircle(A)));
    elif(extentnumber==4):
        finalNeibor=ExtentOneCircle(ExtentOneCircle(ExtentOneCircle(ExtentOneCircle(A))));
    elif (extentnumber == 5):
        finalNeibor=ExtentOneCircle(ExtentOneCircle(ExtentOneCircle(ExtentOneCircle(ExtentOneCircle(A)))))
    elif (extentnumber == 6):
        finalNeibor=ExtentOneCircle(ExtentOneCircle(ExtentOneCircle(ExtentOneCircle(ExtentOneCircle(ExtentOneCircle(A))))))
    return finalNeibor;

##对于多波段，外围扩充1，2，3，4圈
def MubandsWindowSizeEX(Mubands,extentnumber):
    numberBands=Mubands.shape[2];   #shape出通道数
    temp = WindowSizeFunction(Mubands[:,:,0],extentnumber);
    for ii in np.arange(1,numberBands):
        temp = np.dstack((temp, WindowSizeFunction(Mubands[:,:,ii],extentnumber) ));
    output=temp;
    return output;

#----------------------训练和测试样本 Label----------------------------#
#获取每个样本点shape的所有点的坐标,并转为全色影像对应的行列号
'''Osgeo库资料过少看不懂，明确的是输入tiff等和样本点Shp，返回样本点的标签和投影坐标，注意feat.GetField中间参数'''
def GetXYValueToRowsCols(osgeobands,shapePath):
    #获得给定数据的投影参考系和地理参考系
    def getSRSPair(dataset):
        prosrs = osr.SpatialReference()
        prosrs.ImportFromWkt(dataset.GetProjection())
        geosrs = prosrs.CloneGeogCS()
        return prosrs, geosrs;
    #将经纬度坐标转为投影坐标（具体的投影坐标系由给定数据确定）
    def lonlat2geo(dataset, lon, lat):
        prosrs, geosrs = getSRSPair(dataset);
        ct = osr.CoordinateTransformation(geosrs, prosrs);
        coords = ct.TransformPoint(lon, lat);
        return coords[:2];
    #根据GDAL的六参数模型将给定的投影转为影像图上坐标（行列号）
    def geo2imagexy(dataset, x, y):
        trans = dataset.GetGeoTransform();
        a = np.array([[trans[1], trans[2]], [trans[4], trans[5]]]);
        b = np.array([x - trans[0], y - trans[3]]);
        return np.linalg.solve(a, b);  # 使用numpy的linalg.solve进行二元一次方程的求解
    #获取每个样本点shape的所有点的坐标,并转为全色影像对应的行列号
    ogr.RegisterAll();  # 注册所有的驱动
    ds = ogr.Open(shapePath,0);      #以只读模式打开文件
    lyr = ds.GetLayer(0);            #打开shp文件的第一个图层
    DSpan = gdal.Open(osgeobands);        # print(DSpan.GetProjection());
    RowsCols = [];       allLabel = [];
    for feat in lyr:                 #for循环遍历shp第一个图层的所有要素（这里是样点）
        pt = feat.geometry();        #从要素中获得几何形状（即点、线、面），并传递给pt变量
        OneLabel = feat.GetField('DLMC'); #从字段DLMC中获取地类编号，并传递给OneLabel变量
        allLabel.append( int(OneLabel) ); #为避免中文字段造成的困扰，在DLMC中用数值代替中文，为解决小尾数问题，采用int函数
        CoordsXY = lonlat2geo(DSpan,pt.GetX(),pt.GetY());
        tempRowCol = geo2imagexy(DSpan, CoordsXY[0], CoordsXY[1]);
        Row = int(math.floor( tempRowCol[1] ));  Col = int( math.floor( tempRowCol[0] ));
        RowsCols.append( [Row,Col] );
    return RowsCols,np.array(allLabel);

def MakeSample(allRowCols,allLabel, allImageData,EXTnumber):
    #Nubands = allImageData.shape[2];
    afterExtent = MubandsWindowSizeEX(allImageData,EXTnumber);  #原始多波段数据外围进行扩展
    #以邻域窗口进行滑动，获取所有的patch
    #patches = image.extract_patches_2d(afterExtent, (2*EXTnumber+1, 2*EXTnumber+1) );
    #训练样本的patch
    LENTH = len(allRowCols);        #WS = 2*EXTnumber+1;
    #SamplePathes = np.zeros((LENTH,WS, WS,Nubands),dtype = np.uint16);
    SamplePathes = [];              finalUseLabel = [];
    for ii in range(LENTH):
        hang = allRowCols[ii][0]+EXTnumber;    lie = allRowCols[ii][1] + EXTnumber;
        onePath = afterExtent[ hang-EXTnumber:hang+EXTnumber+1, lie-EXTnumber:lie+EXTnumber+1, :];
        if (onePath.shape[0] == onePath.shape[1]):
            SamplePathes.append( onePath );
            finalUseLabel.append( allLabel[ii] );
    return np.array(SamplePathes),  np.array(finalUseLabel)

#根据行列号制作所需的样本(单像素，对于keras的网络框架，输入尺寸为1*8)
'''没看懂'''
def MakeOnePixelSamples(allRowCols,allLabel,allImageData):
    Rows = allImageData.shape[0];   Cols = allImageData.shape[1];   BUb = allImageData.shape[2];
    LENTH = len(allRowCols);   SamplePathes = [];   finalUseLabel = [];#[]申明是列表
    for ii in range(LENTH):
        hang = allRowCols[ii][0];    lie = allRowCols[ii][1] ;
        if (  hang < Rows and lie < Cols ):
            onePixel = allImageData[ hang, lie, :];
            SamplePathes.append( onePixel.reshape(BUb,1) );
            finalUseLabel.append( allLabel[ii] );
    return np.array(SamplePathes),  np.array(finalUseLabel);

def MakeMLPSample(ALLXfeature, allRowCols, Ylable):
    hang = ALLXfeature.shape[0];
    lie = ALLXfeature.shape[1];
    NumberBands = ALLXfeature.shape[2];
    AllXdata = ALLXfeature.reshape(hang * lie, NumberBands);
    SampleIndex = [];
    for temp in allRowCols:
        SampleIndex.append(temp[0] * lie + temp[1]);
    tempX = AllXdata[SampleIndex, :];
    tempY = np.array(Ylable);
    return tempX, tempY;

def MakeConv3dSample(allRowCols,allLabel, allImageData,EXTnumber):
    #Nubands = allImageData.shape[2];
    afterExtent = MubandsWindowSizeEX(allImageData,EXTnumber);  #原始多波段数据外围进行扩展
    #以邻域窗口进行滑动，获取所有的patch
    #patches = image.extract_patches_2d(afterExtent, (2*EXTnumber+1, 2*EXTnumber+1) );
    #训练样本的patch
    LENTH = len(allRowCols);        #WS = 2*EXTnumber+1;
    #SamplePathes = np.zeros((LENTH,WS, WS,Nubands),dtype = np.uint16);
    SamplePathes = [];              finalUseLabel = [];
    for ii in range(LENTH):
        hang = allRowCols[ii][0]+EXTnumber;    lie = allRowCols[ii][1] + EXTnumber;
        onePath = afterExtent[ hang-EXTnumber:hang+EXTnumber+1, lie-EXTnumber:lie+EXTnumber+1, :];
        if (onePath.shape[0] == onePath.shape[1]):
            SamplePathes.append( onePath );
            finalUseLabel.append( allLabel[ii] );
    return np.array(SamplePathes),  np.array(finalUseLabel)

#  样本数据划分 ，按一定比例分出训练集和测试集
def SplitSample(Xdata,Ylabelsample,scale,numberClass):
    X_train,X_test,y_train,y_test = train_test_split (Xdata,Ylabelsample,#分出训练集和验证集
                                    test_size = scale, random_state=2,
                                    stratify = Ylabelsample);   #stratify见简书
    y_trainfinal = keras.utils.to_categorical(y_train-1, num_classes = numberClass);#没看懂-1是什么意思
    y_testfinal = keras.utils.to_categorical(y_test-1, num_classes = numberClass);
    return X_train,X_test,y_trainfinal,y_testfinal;
def SplitSample3D(Xdata,Ylabelsample,scale,numberClass,WS):
    X_train,X_test,y_train,y_test = train_test_split (Xdata,Ylabelsample,#分出训练集和验证集
                                    test_size = scale, random_state=2,
                                    stratify = Ylabelsample);   #stratify见简书
    y_trainfinal = keras.utils.to_categorical(y_train-1, num_classes = numberClass);#没看懂-1是什么意思
    y_testfinal = keras.utils.to_categorical(y_test-1, num_classes = numberClass);
    X_train = X_train.reshape(X_train.shape[0],WS,WS,30,1)
    X_test = X_test.reshape(X_test.shape[0],WS,WS,30,1)
    return X_train,X_test,y_trainfinal,y_testfinal;

def MLPSplitSample(Xdata, Ylabelsample, scale):
    X_train, X_test, y_train, y_test = train_test_split(Xdata, Ylabelsample,
                                                        test_size=scale, random_state=2,
                                                        stratify=Ylabelsample);
    return X_train, X_test, y_train, y_test;


#多类别分类结果绘图
def Draw2orMoreClassResult(PYLabel,titleName):
    ColorList = ['Yellow','Black'];
    # 样本数据，1为水田，2为旱地，3为建筑，4为水体，5为林地，6为园地，7为其他地类
    # TH7color = ['Orange','Yellow','Red', 'Blue', 'Green', 'Gray', 'Black'];
    cor = ListedColormap(ColorList);
    plt.figure(figsize=(20,20));
    plt.imshow(PYLabel,cmap=cor);
    plt.xticks(fontsize=20);   plt.yticks(fontsize=20);
    #plt.colorbar();
    plt.title(titleName,fontsize = 30);
