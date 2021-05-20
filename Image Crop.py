import math
import numpy as np
import os
import cv2 as cv
from numpy.core.defchararray import array

def Load_Dict(img_perfix,anno_perfix,filename):
    '''
    args:
        img_perfix,anno_perfix: path of folder
        filename: file name without suffix
    return:
        img: array (height,width,channel)
        annoatations: list[list[...]] (str)
    '''
    
    img_file=os.path.join(img_perfix,filename+'.jpg')
    anno_file=os.path.join(anno_perfix,filename+'.txt')

    img=cv.imread(img_file)
    with open(anno_file,'rt') as f:
        annotations=[]
        for line in f.readlines():
            annotations.append(line.replace('\n','').split(','))
        # convert to array-like data
        annotations=np.array(annotations,dtype=float)
        # set label of ignored region to -1
        annotations[annotations[:,4]==0]=-1
    return img,np.array(annotations,dtype=float)


def Crop(img,annotation,height=640,width=640,select_mode='inside'):

    ### What's 'mean' means
    mean = [123.675, 116.28, 103.53]
    mean_ = mean[ : :-1]

    h, w, c=img.shape

    num_h=math.ceil(h/height)
    num_w=math.ceil(w/width)
    
    # generate crop beginning coordinate
    h_begin=[height*i for i in range(num_h)]
    h_end=[height*(i+1) for i in range(num_h)]
    w_begin=[width*i for i in range(num_w)]
    w_end=[width*(i+1) for i in range(num_w)]

    # ensure the crop within origin img
    h_begin[-1]= max(h-height,0)
    w_begin[-1]=max(w-width,0)

    h_end[-1]=h
    w_end[-1]=w

    new_imgs=[]
    new_annos=[]
    for i in range(num_h):
        current_h=h_end[i]-h_begin[i]

        for j in range(num_w):
            current_w=w_end[j]-w_begin[j]
            new_img=np.full((height,width,c),mean_,
                            dtype=img.dtype)
            new_img[:current_h,:current_w]=img[h_begin[i]:h_end[i]
                                                ,w_begin[j]:w_end[j]]
            
            new_imgs.append(new_img)

            if select_mode == 'inside':
                current_anno=annotation
                
                # relocation object into crop img
                # current_anno : [x_left,y_top,w,h,...]
                current_anno[:,0] -= w_begin[j]
                current_anno[:,1] -= h_begin[i]

                x_right=current_anno[:,0]+current_anno[:,2]
                y_bottome=current_anno[:,1]+current_anno[:,3]
                
                # coordinate base on new crop img 
                current_anno[current_anno[:,0]<0,0]=0
                current_anno[current_anno[:,1]<0,1]=0
                x_right[x_right[:]>width]=width
                y_bottome[y_bottome[:]>height]=height

                current_anno_w=x_right-current_anno[:,0]
                current_anno_h=y_bottome-current_anno[:,1]
                
                # choose bbox intersect with new img 
                index=np.where((current_anno_w>0) & (current_anno_h>0))
                current_anno=current_anno[index]
                current_anno_w=current_anno_w[index]
                current_anno_h=current_anno_h[index]
                # resize width & height of bbox
                current_anno[:,2]=current_anno_w
                current_anno[:,3]=current_anno_h

                if current_anno.shape[0]==0:
                    current_anno=np.zeros((1,8))
                new_annos.append(current_anno)

    return new_imgs,new_annos


if __name__=='__main__':

    IMG_PERFIX='../VisDrone2019-DET-train/images'
    ANNO_PERFIX='../VisDrone2019-DET-train/annotations'
    SAVE_PERFIX='../VisDrone2019-DET-train-crop'
    TOTOAL_NUM=0

    files=os.listdir(IMG_PERFIX)

    for i,file in enumerate(files):
        
        filename=file.split('.')[0]

        img,annotations=Load_Dict(IMG_PERFIX,ANNO_PERFIX,filename)
        new_img,new_annotations=Crop(img,annotations,height=412,width=412)

        TOTOAL_NUM+=len(new_img)

        for j in range(len(new_img)):
            img_path=os.path.join(SAVE_PERFIX,filename+'_%d.jpg'%i)
            anno_path=os.path.join(SAVE_PERFIX,filename+'_%d.txt'%i)

            # write new crop img
            #cv.imwrite(img_path,new_img[i])

            # write new annotation files in VisDrone .txt form  
            #ignore_index=(new_annotations[j,:,4]==-1)
            ignore_index=[0,1]
            l=new_annotations[j,~ignore_index,4]
            a=1

                
    
