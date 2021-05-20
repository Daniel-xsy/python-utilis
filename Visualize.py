##########################################################
# 
# This file visualize bbox from coco-like annotataion file
#
###########################################################

import cv2 as cv
import os
import pickle

img_prefix='E:/data/patch/images/'
img_range=[0,76774]
annotation_path='E:/data/patch/pkl_files/patch_train.pkl'
save=False
save_path='E:/data/patch/visualization/'
classes=['pedestrian','people','bicycle','car','van','truck','tricycle','awning-tricycle','bus','motor','others']


def Visualize(img_prefix,img_range,annotation_path,save=False,save_path='./'):
    # img_range [begin,end]

    begin=img_range[0]
    end=img_range[1]
    num=end-begin

    print('loading data...')
    anno_file=open(annotation_path,'rb+')
    info=pickle.load(anno_file)
    anno_file.close()
    annotations=info[begin:end]


    for i,anno in enumerate(annotations):
        if i%100 ==0:
            print('processing {}/{}....'.format(i+1,num))
        img_path=os.path.join(img_prefix,anno['filename'])
        img=cv.imread(img_path)

        bboxes=anno['ann']['bboxes']
        for j,bbox in enumerate(bboxes):
            p_left_top=(bbox[0],bbox[1])
            p_right_bot=(bbox[2],bbox[3])
            img=cv.rectangle(img,p_left_top,p_right_bot,(0,255,0),thickness=1)

        save_file=os.path.join(save_path,anno['filename'])
        cv.imwrite(save_file,img)
    print('Done!')


if __name__=='__main__':

    Visualize(img_prefix=img_prefix,img_range=img_range,
                annotation_path=annotation_path,save=save,save_path=save_path)