'''

pkl is a list,its length is the number of images.
each element is a dict including {'filename', 'width', 'height', 'ann'}.
'annot' is a dict including {'bboxes', 'labels', 'bboxes_ignore', 'labels_ignore'}.
box's dim is N*4,[x1, y1, x2, y2].

$$$$$important!!!label from 1 !!!$$$$$
Example:
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


Example:
'images': [
    {
        'file_name': 'COCO_val2014_000000001268.jpg',
        'height': 427,
        'width': 640,
        'id': 1268
    },
    ...
],

'annotations': [
    {
        'segmentation': [[192.81,
            247.09,
            ...
            219.03,
            249.06]],  # if you have mask labels
        'area': 1035.749,
        'iscrowd': 0, ##if ==1 it's be ignored
        'image_id': 1268,
        'bbox': [192.81, 224.8, 74.73, 33.43],  ###[x1, y1, w, h]
        'category_id': 16,
        'id': 42986
    },
    ...
],

'categories': [
    {'id': 0, 'name': 'car'},
 ]
'''

import os
import numpy as np
import pickle as pickle
import json

def clip_bbox(bbox, height, width):
    bbox[bbox[:,0]<0,0]=0
    bbox[bbox[:,1]<0,1]=0
    bbox[bbox[:,2]>width,2]=width
    bbox[bbox[:,3]>height,3]=height
    return list(bbox)
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, time):
            return obj.__str__()
        else:
            return super(NpEncoder, self).default(obj)

pkl_file = '../patch/patch_train.pkl'
json_path = '../patch_train_coco.json'
fr = open(pkl_file, "rb+")
inf = pickle.load(fr)
fr.close()

total_num = len(inf)

image_id = 1
bbox_id = 1
json_dict = {'images':[], 'annotations': [], 'categories': []}
cats = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
for cid, cate in enumerate(cats):
        cat = {'id': cid, 'name': cate}
        json_dict['categories'].append(cat)

for num, annot_dict in enumerate(inf):
    if num%100==0:
        print('processing %d/%d'%(num, total_num), end='\r')
     
    annotation = {}
    categories = {} 
    ##read info
    file_name = annot_dict['filename']
    width = annot_dict['width']
    height = annot_dict['height']
    bboxes = annot_dict['ann']['bboxes']
    labels = annot_dict['ann']['labels']
    bboxes_ignore = annot_dict['ann']['bboxes_ignore']
    bboxes = clip_bbox(bboxes, height, width)
    bboxes_ignore = clip_bbox(bboxes_ignore, height, width)
    ##image_dict
    image = {}
    image['file_name'] = file_name
    image['height'] = height
    image['width'] = width
    image['id'] = image_id
    json_dict['images'].append(image)

    ##annotations
    ##ignore
    assert len(bboxes) == len(labels)
    for j in range(len(bboxes_ignore)):
        gt_box = bboxes_ignore[j]
        fbox = [gt_box[0], gt_box[1], gt_box[2]-gt_box[0], gt_box[3]-gt_box[1]]
        annotation = {'area': fbox[2]*fbox[3], 'iscrowd': 1, 'image_id':image_id, 'bbox':fbox, 
                    'category_id': -1,'id': bbox_id,'segmentation': []} 
        json_dict['annotations'].append(annotation) 
        bbox_id += 1
    ##gt
    for j in range(len(bboxes)):
        category_id = labels[j] - 1
        if category_id<0 or category_id>9:
            raise print('category_id is error')
        gt_box = bboxes[j]
        fbox = [gt_box[0], gt_box[1], gt_box[2]-gt_box[0], gt_box[3]-gt_box[1]]
        annotation = {'area': fbox[2]*fbox[3], 'iscrowd': 0, 'image_id':image_id, 'bbox':fbox, 
                    'category_id': category_id,'id': bbox_id,'segmentation': []}  
        json_dict['annotations'].append(annotation) 
        bbox_id += 1
    image_id += 1 


print('writing file...')
json_fp = open(json_path, 'w', encoding='utf-8')
json_str = json.dumps(json_dict,cls=MyEncoder)
json_fp.write(json_str)
json_fp.close()
