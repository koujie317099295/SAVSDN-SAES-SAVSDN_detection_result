import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor

def getNormalizeImg(img,lenNormalize):
    height=img.shape[0]
    width=img.shape[1]
    
    if height>=width:
        newWidth=int(width*lenNormalize/height)
        img = cv2.resize(img, (newWidth, lenNormalize))
    else:
        newHeight=int(height*lenNormalize/width)
        img = cv2.resize(img, (lenNormalize, newHeight))
        
    return img

def generateTrainData(mp4,picIdx):
    #初始化
    lenNormalize=1024

    #开始操作
    cap = cv2.VideoCapture(mp4)
    ret, frame = cap.read()
    while ret:
        ret, frame = cap.read()
        if ret:
            #初始化
            valFlag=np.random.randint(0,1000,1)[0]#确定该次数据为训练数据还是测试数据 训练数据和测试数据比例为999:1 防止大量数据被浪费在测试部分

            #开始操作
            frame=getNormalizeImg(frame,lenNormalize)#图像尺寸归一化
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#RGB-->GREY单通道
            
            temp=np.concatenate((frame[:,:,None],frame[:,:,None]),axis=2)
            yuChuLiResult=np.concatenate((temp,frame[:,:,None]),axis=2)#GREY单通道-->GREY三通道     

            if valFlag==-1:#保存添加或者未添加火星后的图片
                cv2.imwrite('data/images/624_mySparkImgsLabelsXXXXXX{}.jpg'.format(picIdx), yuChuLiResult)
            else:
                cv2.imwrite('data/images/624_mySparkImgsLabelsXXXXXX{}.jpg'.format(picIdx), yuChuLiResult)
            cv2.imshow('tianJiaHuoXing', yuChuLiResult)#展示添加火星后图像
            
            #更新状态
            picIdx+=1
            
            #结束
            k = cv2.waitKey(1) & 0xff#这里可以设置图像展示的最高帧率（实际帧率有可能被处理速度拖下来）
            if k == 27:
                break
        else:
            break
    
    cv2.destroyAllWindows()
    cap.release()
    
    return picIdx

def mp4s2imgs():
    #初始化
    filepath = 'videos' #视频文件所在目录
    pathDir = os.listdir(filepath)
    picIdx=1
    videoPathsList=[]
    videoFramesNumsList=[]
    tasks=[]
    for dir in pathDir:
        videoPath ='videos/'+dir
        cap=cv2.VideoCapture(videoPath)
        frames_num=cap.get(7)
        videoPathsList.append(videoPath)
        videoFramesNumsList.append(frames_num)
        tasks.append('')

    #开始操作
    framsAll=0 
    print('累计视频帧数统计如下：')
    for i in videoFramesNumsList:
        framsAll+=i
        print(framsAll)
        

    for i in range(0,len(tasks)):
        print('正在处理视频"{}"中...'.format(videoPathsList[i]))
        picIdx=generateTrainData(videoPathsList[i],picIdx)
        print('视频"{}"已经处理完毕！'.format(videoPathsList[i]))

    return picIdx

