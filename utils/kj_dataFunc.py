import math
import os
import random
import numpy as np
import cv2 as cv
from utils.datasets import ImgAndLabels,Imgs
from concurrent.futures import ThreadPoolExecutor
import torch

def get9ImagesAndLabels(paths,imgSize,numOfFrames):   
    #初始化一组数据增广随机参数
    flip_1 , flip_2 = random.random(),random.random()
    mydegrees,mytranslate,myscale,myshear,myperspective,myborder=0.0,0.1,0.5,0.0,0.0,(-imgSize//2, -imgSize//2) 
    myheight=imgSize*2 + myborder[0] * 2
    mywidth=imgSize*2 + myborder[1] * 2

    r=[]
    r.append(random.uniform(-myperspective, myperspective))
    r.append(random.uniform(-myperspective, myperspective))
    r.append(random.uniform(-mydegrees, mydegrees))
    r.append(random.uniform(1 - myscale, 1 + myscale))
    r.append(math.tan(random.uniform(-myshear, myshear) * math.pi / 180))
    r.append(math.tan(random.uniform(-myshear, myshear) * math.pi / 180))
    r.append(random.uniform(0.5 - mytranslate, 0.5 + mytranslate) * mywidth)
    r.append(random.uniform(0.5 - mytranslate, 0.5 + mytranslate) * myheight)
    r.append(random.randint(0,imgSize))
    r.append(random.randint(0,imgSize))
    #初始化数据索引 
    #7、8、9帧数据索引准备
    strlist = paths[0].split('XXXXXX')	#用逗号分割str字符串，并保存到列表
    videoName=strlist[0]
    
    temp=int(strlist[1][:-4])    
    if temp<=10 or temp>=numOfFrames-10:
        print('正在重新选取训练数据')
        temp=np.random.randint(10,numOfFrames-10,1)[0]
    prePicIdx,picIdx,nextPicIdx=str(temp-1),str(temp),str(temp+1)
    
    prePicPath,PicPath,nextPicPath=videoName+'XXXXXX'+prePicIdx+'.jpg', videoName+'XXXXXX'+picIdx+'.jpg', videoName+'XXXXXX'+nextPicIdx+'.jpg'
    preLabelPath,picLabelPath,nextLabelPath=videoName.replace('images','labels')+'XXXXXX'+prePicIdx+'.txt',videoName.replace('images','labels')+'XXXXXX'+picIdx+'.txt',videoName.replace('images','labels')+'XXXXXX'+nextPicIdx+'.txt'

    #1~6帧数据索引准备
    PicIdxf7 ,picIdxf6 ,picIdxf5 ,picIdxf4 ,picIdxf3 ,picIdxf2=str(temp-7) ,str(temp-6) ,str(temp-5) ,str(temp-4) ,str(temp-3) ,str(temp-2)
    PicPathf7 ,PicPathf6 ,PicPathf5 ,PicPathf4 ,PicPathf3 ,PicPathf2=videoName+'XXXXXX'+PicIdxf7+'.jpg' ,videoName+'XXXXXX'+picIdxf6+'.jpg' ,videoName+'XXXXXX'+picIdxf5+'.jpg' ,videoName+'XXXXXX'+picIdxf4+'.jpg' ,videoName+'XXXXXX'+picIdxf3+'.jpg' ,videoName+'XXXXXX'+picIdxf2+'.jpg'
    LabelPathf7=r'..\624_mySparkImgsLabels\labels\blank.txt'
    LabelPathf6,LabelPathf5,LabelPathf4,LabelPathf3,LabelPathf2= LabelPathf7,LabelPathf7,LabelPathf7,LabelPathf7,LabelPathf7   

    #使用线程池进行训练数据的准备
    #系统启动一个新线程的成本是比较高的，因为它涉及与操作系统的交互。在这种情形下，使用线程池可以很好地提升性能，尤其是当程序中需要创建大量生存期很短暂的线程时，更应该考虑使用线程池。
    #线程池在系统启动时即创建大量空闲的线程，程序只要将一个函数提交给线程池，线程池就会启动一个空闲的线程来执行它。当该函数执行结束后，该线程并不会死亡，而是再次返回到线程池中变成空闲状态，等待执行下一个函数。
    #此外，使用线程池可以有效地控制系统中并发线程的数量。当系统中包含有大量的并发线程时，会导致系统性能急剧下降，甚至导致 Python 解释器崩溃，而线程池的最大线程数参数可以控制系统中并发线程的数量不超过此数。
    #上面程序中，第 13 行代码创建了一个包含两个线程的线程池，接下来的两行代码只要将 action() 函数提交（submit）给线程池，该线程池就会负责启动线程来执行 action() 函数。这种启动线程的方法既优雅，又具有更高的效率。
    #当程序使用 Future 的 result() 方法来获取结果时，该方法会阻塞当前线程，如果没有指定 timeout 参数，当前线程将一直处于阻塞状态，直到 Future 代表的任务返回。
    def action(imgPath,labelPath,flip_up_down,flip_left_right,randomList):
        imgAndLabels=ImgAndLabels(imgPath=imgPath,labelPath=labelPath, img_size=imgSize, flip_up_down=flip_up_down , flip_left_right=flip_left_right ,randomList=randomList,random_perspective=True)
        otherImgInput,LabelInput=imgAndLabels.get()
        return otherImgInput,LabelInput
    # 创建一个包含2条线程的线程池
    pool = ThreadPoolExecutor(max_workers=13)
    # 向线程池提交task
    future0 = pool.submit(action,PicPathf7,LabelPathf7,flip_1,flip_2,r)
    future1 = pool.submit(action,PicPathf6,LabelPathf6,flip_1,flip_2,r)
    future2 = pool.submit(action,PicPathf5,LabelPathf5,flip_1,flip_2,r)
    future3 = pool.submit(action,PicPathf4,LabelPathf4,flip_1,flip_2,r)
    future4 = pool.submit(action,PicPathf3,LabelPathf3,flip_1,flip_2,r)
    future5 = pool.submit(action,PicPathf2,LabelPathf2,flip_1,flip_2,r)
    future6 = pool.submit(action,prePicPath,preLabelPath,flip_1,flip_2,r)
    future7 = pool.submit(action,PicPath,picLabelPath,flip_1,flip_2,r)
    future8 = pool.submit(action,nextPicPath,nextLabelPath,flip_1,flip_2,r)
    # 获取各任务返回的结果
    otherImgInput0,_=future0.result()
    otherImgInput1,_=future1.result()
    otherImgInput2,_=future2.result()
    otherImgInput3,_=future3.result()
    otherImgInput4,_=future4.result()
    otherImgInput5,_=future5.result()
    otherImgInput6,LabelInput6=future6.result()
    otherImgInput7,LabelInput7=future7.result()
    otherImgInput8,LabelInput8=future8.result()
    # 关闭线程池
    pool.shutdown()
    #获取1~9帧图像输入
    otherImgInput=torch.cat([otherImgInput0,otherImgInput1],dim=2)
    otherImgInput=torch.cat([otherImgInput,otherImgInput2],dim=2)
    otherImgInput=torch.cat([otherImgInput,otherImgInput3],dim=2)
    otherImgInput=torch.cat([otherImgInput,otherImgInput4],dim=2)
    otherImgInput=torch.cat([otherImgInput,otherImgInput5],dim=2)
    otherImgInput=torch.cat([otherImgInput,otherImgInput6],dim=2)
    otherImgInput=torch.cat([otherImgInput,otherImgInput7],dim=2)
    otherImgInput=torch.cat([otherImgInput,otherImgInput8],dim=2)
    #获取7、8、9帧图像输入对应标签
    labelInput=torch.cat([LabelInput6,LabelInput7],dim=0)
    labelInput=torch.cat([labelInput,LabelInput8],dim=0)
    #单独获取7、8、9帧图像输入
    imgInput0=torch.squeeze(otherImgInput6,0)
    imgInput0=torch.squeeze(imgInput0,1)
    imgInput1=torch.squeeze(otherImgInput7,0)
    imgInput1=torch.squeeze(imgInput1,1)
    imgInput2=torch.squeeze(otherImgInput8,0)
    imgInput2=torch.squeeze(imgInput2,1)

    imgInput0=torch.transpose(imgInput0,0,2)
    imgInput0=torch.transpose(imgInput0,0,1)
    imgInput1=torch.transpose(imgInput1,0,2)
    imgInput1=torch.transpose(imgInput1,0,1)
    imgInput2=torch.transpose(imgInput2,0,2)
    imgInput2=torch.transpose(imgInput2,0,1)
    
    imgInput012=(imgInput0/1.0+imgInput1/1.0+imgInput2/1.0)/3
    imgInput012=np.array(imgInput012.numpy(),dtype='uint8')

    imgInput012=torch.from_numpy(imgInput012)#H W C
    imgInput012=torch.transpose(imgInput012,0,2)#C W H
    imgInput012=torch.transpose(imgInput012,1,2)#C H W
    imgInput012=torch.unsqueeze(imgInput012,0)#B C H W
        
    return otherImgInput,labelInput,imgInput012
    
def get9Images(path,imgSize,framsAll):   
    #初始化数据索引 
    #7、8、9帧数据索引准备
    strlist = path.split('XXXXXX')	#用逗号分割str字符串，并保存到列表
    videoName=strlist[0]
    temp=int(strlist[1][:-4])    

    if temp<=10 or temp>=framsAll-10:
        print('正在重新选取训练数据')
        temp=np.random.randint(10,framsAll-10,1)[0]           
        
    prePicIdx,picIdx,nextPicIdx=str(temp-1),str(temp),str(temp+1)
    
    prePicPath,PicPath,nextPicPath=videoName+'XXXXXX'+prePicIdx+'.jpg', videoName+'XXXXXX'+picIdx+'.jpg', videoName+'XXXXXX'+nextPicIdx+'.jpg'

    #1~6帧数据索引准备
    PicIdxf7 ,picIdxf6 ,picIdxf5 ,picIdxf4 ,picIdxf3 ,picIdxf2=str(temp-7) ,str(temp-6) ,str(temp-5) ,str(temp-4) ,str(temp-3) ,str(temp-2)
    PicPathf7 ,PicPathf6 ,PicPathf5 ,PicPathf4 ,PicPathf3 ,PicPathf2=videoName+'XXXXXX'+PicIdxf7+'.jpg' ,videoName+'XXXXXX'+picIdxf6+'.jpg' ,videoName+'XXXXXX'+picIdxf5+'.jpg' ,videoName+'XXXXXX'+picIdxf4+'.jpg' ,videoName+'XXXXXX'+picIdxf3+'.jpg' ,videoName+'XXXXXX'+picIdxf2+'.jpg'

    #使用线程池进行训练数据的准备
    #系统启动一个新线程的成本是比较高的，因为它涉及与操作系统的交互。在这种情形下，使用线程池可以很好地提升性能，尤其是当程序中需要创建大量生存期很短暂的线程时，更应该考虑使用线程池。
    #线程池在系统启动时即创建大量空闲的线程，程序只要将一个函数提交给线程池，线程池就会启动一个空闲的线程来执行它。当该函数执行结束后，该线程并不会死亡，而是再次返回到线程池中变成空闲状态，等待执行下一个函数。
    #此外，使用线程池可以有效地控制系统中并发线程的数量。当系统中包含有大量的并发线程时，会导致系统性能急剧下降，甚至导致 Python 解释器崩溃，而线程池的最大线程数参数可以控制系统中并发线程的数量不超过此数。
    #上面程序中，第 13 行代码创建了一个包含两个线程的线程池，接下来的两行代码只要将 action() 函数提交（submit）给线程池，该线程池就会负责启动线程来执行 action() 函数。这种启动线程的方法既优雅，又具有更高的效率。
    #当程序使用 Future 的 result() 方法来获取结果时，该方法会阻塞当前线程，如果没有指定 timeout 参数，当前线程将一直处于阻塞状态，直到 Future 代表的任务返回。
    def action(imgPath):
        imgs=Imgs(imgPath=imgPath, img_size=imgSize)
        ImgInput=imgs.get()
        return ImgInput
    # 创建一个包含13条线程的线程池
    pool = ThreadPoolExecutor(max_workers=13)
    # 向线程池提交task
    future0 = pool.submit(action,PicPathf7)
    future1 = pool.submit(action,PicPathf6)
    future2 = pool.submit(action,PicPathf5)
    future3 = pool.submit(action,PicPathf4)
    future4 = pool.submit(action,PicPathf3)
    future5 = pool.submit(action,PicPathf2)
    future6 = pool.submit(action,prePicPath)
    future7 = pool.submit(action,PicPath)
    future8 = pool.submit(action,nextPicPath)
    # 获取各任务返回的结果
    otherImgInput0=future0.result()#B C T H W
    otherImgInput1=future1.result()
    otherImgInput2=future2.result()
    otherImgInput3=future3.result()
    otherImgInput4=future4.result()
    otherImgInput5=future5.result()
    otherImgInput6=future6.result()
    otherImgInput7=future7.result()
    otherImgInput8=future8.result()
    # 关闭线程池
    pool.shutdown()
    #获取1~9帧图像输入 tensor
    otherImgInput=torch.cat([otherImgInput0,otherImgInput1],dim=2)
    otherImgInput=torch.cat([otherImgInput,otherImgInput2],dim=2)
    otherImgInput=torch.cat([otherImgInput,otherImgInput3],dim=2)
    otherImgInput=torch.cat([otherImgInput,otherImgInput4],dim=2)
    otherImgInput=torch.cat([otherImgInput,otherImgInput5],dim=2)
    otherImgInput=torch.cat([otherImgInput,otherImgInput6],dim=2)
    otherImgInput=torch.cat([otherImgInput,otherImgInput7],dim=2)
    otherImgInput=torch.cat([otherImgInput,otherImgInput8],dim=2)
    #单独获取7、8、9帧图像输入 numpy
    imgInput0=torch.squeeze(otherImgInput6,0)
    imgInput0=torch.squeeze(imgInput0,1)
    imgInput1=torch.squeeze(otherImgInput7,0)
    imgInput1=torch.squeeze(imgInput1,1)
    imgInput2=torch.squeeze(otherImgInput8,0)
    imgInput2=torch.squeeze(imgInput2,1)

    imgInput0=torch.transpose(imgInput0,0,2)
    imgInput0=torch.transpose(imgInput0,0,1)
    imgInput1=torch.transpose(imgInput1,0,2)
    imgInput1=torch.transpose(imgInput1,0,1)
    imgInput2=torch.transpose(imgInput2,0,2)
    imgInput2=torch.transpose(imgInput2,0,1)
    
    imgInput012=(imgInput0/1.0+imgInput1/1.0+imgInput2/1.0)/3
    imgInput012=np.array(imgInput012.numpy(),dtype='uint8')#H W C

    return otherImgInput,imgInput012
    
