import cv2
import glob
import os

def getIdx(path):
    strlist = path.split('XXXXXX')	#用逗号分割str字符串，并保存到列表

    return int(strlist[1][:-4])
 
def resize(img_array, align_mode):
    _height = len(img_array[0])
    _width = len(img_array[0][0])
    for i in range(1, len(img_array)):
        img = img_array[i]
        height = len(img)
        width = len(img[0])
        if align_mode == 'smallest':
            if height < _height:
                _height = height
            if width < _width:
                _width = width
        else:
            if height > _height:
                _height = height
            if width > _width:
                _width = width
 
    for i in range(0, len(img_array)):
        img1 = cv2.resize(img_array[i], (_width, _height), interpolation=cv2.INTER_CUBIC)
        img_array[i] = img1
 
    return img_array, (_width, _height)
 
def images_to_video(path):
    img_array = []
        
    file_dir = "runs/detect/exp"    
    for root, dirs, files in os.walk(file_dir):  
        paths=files
    #提取名字list中的序号
    idxsList=[]
    for path in paths:
        idxsList.append(getIdx(path))
    #将序号从小到大排序
    idxsList.sort()
    #开始添加顺序的图像序列
    for i in idxsList:
        srcFile = 'runs/detect/exp/624_mySparkImgsLabelsXXXXXX{}.jpg'.format(i)
        img = cv2.imread(srcFile)
        img_array.append(img)
    # 图片的大小需要一致
    img_array, size = resize(img_array, 'largest')
    fps = 30
    out = cv2.VideoWriter('runs/detect/exp/demo.mp4', cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
 
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
 
if __name__ == "__main__":
    images_to_video("runs/detect/exp/")