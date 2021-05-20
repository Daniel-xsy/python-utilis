'''

These File transform VisDrone Dataset annoation forms to pkl file

pkl Example:
{'filename': '0000351_03529_d_0000537_5.jpg', 
'width': 640, 
'height': 640, 
'ann': {'bboxes': array([[532., 331., 573., 418.],
	   [509., 354., 537., 404.],
	   [571.,  -8., 584.,  14.],
	   [550.,  35., 586.,  57.],
	   [559.,  25., 592.,  51.]], dtype=float32), 
	   'labels': array([ 1,  1,  2,  3, 10], dtype=int64), 
	   'bboxes_ignore': array([], shape=(0, 4), dtype=float32), 
	   'labels_ignore': array([], dtype=int64)}}

'''



from __future__ import annotations
import pickle
import os 
import numpy as np
import cv2 as cv

IMG_PREFIX='../VisDrone2019-DET-train/images'
ANNO_PERFIX='../VisDrone2019-DET-train/annotations'
SAVE_PATH='../VisDrone2019-DET-train.pkl'

anno_files=os.listdir(ANNO_PERFIX)
length=len(anno_files)

info=[]
for i,file in enumerate(anno_files):

    file=file.split('.')[0]

    bboxes=[]
    labels=[]
    bboxes_ignore=[]
    labels_ignore=[]

    img_path=os.path.join(IMG_PREFIX,file+'.jpg')
    anno_path=os.path.join(ANNO_PERFIX,file+'.txt')

    img=cv.imread(img_path)
    height,width=img.shape[:2]

    with open(anno_path) as f:

        for line in f.readlines():
            line=np.array(line.strip('\n').split(',')[0:8],dtype=float)
    
            if line[4] != 0:
                bbox=[line[0],line[1],line[0]+line[2],line[1]+line[3]]
                label=line[4]   

                bboxes.append(bbox)
                labels.append(label)
            ## ignored region
            else:
                bbox_ignore=[line[0],line[1],line[0]+line[2],line[1]+line[3]]
                label_ignore=-1

                bboxes_ignore.append(bbox_ignore)
                labels_ignore.append(label_ignore)

        if len(labels_ignore)==0:
            bbox_ignore.append([0,0,0,0])
            labels_ignore.append(-1)

        ann_dict={'bboxes': np.array(bboxes,dtype='float32'),
                  'labels': np.array(labels,dtype='int64'),
                  'bboxes_ignore': np.array(bboxes_ignore,dtype='float32'),
                  'labels_ignore': np.array(labels_ignore,dtype='int64') 
                 }


        dict_ = {'filename': file+'.jpg',
                 'width': width,
                 'height': height,
                 'ann': ann_dict    
                 }

        info.append(dict_)

    if i % 100 ==0:
        print('processing {} / {} ...'.format(i,length))

outfile=open(SAVE_PATH,'wb')
pickle.dump(info,outfile)
    


