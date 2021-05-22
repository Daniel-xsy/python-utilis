import cv2 as cv
import os 

img_prefix='../Version1/images/'

def Img_Size(img_prefix):
    files=os.listdir(img_prefix)
    sizes=dict()
    total_num=len(files)

    for i,file in enumerate(files):
        if i % 10 == 0 :
            print('processing {}/{}'.format(i,total_num))
        img_path=os.path.join(img_prefix,file)
        img=cv.imread(img_path)
        size=str(img.shape[0])+'x'+str(img.shape[1])
        if size not in sizes.keys():
            sizes[size]=0
        sizes[size]+=1  
    
    return sizes


if __name__=='__main__':
    result=Img_Size(img_prefix)
    for key in result.keys():
        print('{} : {}'.format(key,result[key]))

