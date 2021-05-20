##########################################################
# 
# This file visualize bbox from coco-like annotataion file
#
###########################################################

import cv2 as cv
import os
import json

img_prefix='E:/data/patch/images/'
img_range=[0,1]
annotation_path='E:/data/patch/patch_train_coco.json'
save=False
save_path='E:/data/patch/visualization/'

def Visualize(img_prefix,img_range,annotation_path,save=False,save_path='./'):
    # img_range [begin,end]
    anno_file=open(annotation_path,'r')
    info=json.loads(anno_file.read())
    
    anno=info['annotations'][:]['image_id'] in range(img_range[0],img_range[1])

    files=os.listdir(img_prefix)
    imgs_name=files[img_range[0]:img_range[1]]

    for i,img_name in enumerate(imgs_name):
        img_path=os.path.join(img_prefix,img_name)
        img=cv.imread(img_path)


    a=1

if __name__=='__main__':

    Visualize(img_prefix=img_prefix,img_range=img_range,
                annotation_path=annotation_path,save=save,save_path=save_path)