import numpy as np
import os
import math
import cv2
import pickle
'''
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

def img_patch(img_prefix, annot_dict, patch_w=640, patch_h=640, save_img=True):
	##padding means
	mean_ = [123.675, 116.28, 103.53]
	mean_ = mean_[::-1]
	##get basic info
	file_name = os.path.join(img_prefix, annot_dict['filename'])
	if save_img:
		img = cv2.imread(file_name)
	w = annot_dict['width']
	h = annot_dict['height']
	if save_img:
		img_h, img_w, img_c = img.shape
		if img_h!=h or img_w!=w:
			raise print('h or w error')
	
	##get annot and concat [x1, y1, x2, y2, class_num]
	##ignore_class set -1
	annots_tmp = annot_dict['ann']
	annots_tmp['labels_ignore'][:] = -1
	annots = np.concatenate((annots_tmp['bboxes'], annots_tmp['labels'][:,np.newaxis]),axis=1)
	
	if annots_tmp['bboxes_ignore'].shape[0]!=0:
		ignore_annots = np.concatenate(
			(annots_tmp['bboxes_ignore'], annots_tmp['labels_ignore'][:,np.newaxis]),axis=1)
		annots = np.concatenate((annots, ignore_annots),axis=0)
	
	##get the number of patch and index of each patch begining
	num_w = math.ceil(w / patch_w)
	num_h = math.ceil(h / patch_h)
	each_patch_begin_w = [x*patch_w for x in range(num_w)]
	each_patch_begin_h = [x*patch_h for x in range(num_h)]
	each_patch_begin_w[-1] = w - patch_w if (w-patch_w>=0) else 0
	each_patch_begin_h[-1] = h - patch_h if (h-patch_h>=0) else 0
	
	# patch the image
	patch_annots = []
	patch_img = []
	for i in range(num_w):
		begin_w = each_patch_begin_w[i]
		len_w = min(patch_w, w - begin_w)
		end_w = begin_w + len_w
		for j in range(num_h):
			begin_h = each_patch_begin_h[j]
			len_h = min(patch_h, h - begin_h)
			end_h = begin_h + len_h
			if save_img:
				new_image = np.full((patch_h, patch_w, img_c),
							 mean_,
							 dtype=img.dtype)
				new_image[:len_h, :len_w] = img[begin_h:end_h, begin_w:end_w]
				patch_img.append(new_image)

				# new_image = np.zeros((patch_h, patch_w, img_c)).astype(img.dtype)
				# new_image[:len_h, :len_w, :] = img[begin_h:end_h, begin_w:end_w, :]
				# patch_img.append(new_image)

			## patch the annotations
			if annots.shape[0] == 0:
				patch_annots.append(np.zeros((0,5)))
				continue
			current_an = annots.copy()
			current_an[:,[0,2]] -= begin_w
			current_an[:,[1,3]] -= begin_h

			##method one
			center_w = (current_an[:,0] + current_an[:,2])/2
			center_h = (current_an[:,1] + current_an[:,3])/2
			select_index = np.where((center_h>=0) & (center_h<=len_w) & (center_w>=0) & (center_w<=len_w))
			current_an = current_an[select_index]
			
			##clip bbox into image_size and grid of area=0
			# current_an[current_an[:,0]<0,0]=0
			# current_an[current_an[:,1]<0,1]=0
			# current_an[current_an[:,2]>patch_w,2]=patch_w
			# current_an[current_an[:,3]>patch_h,3]=patch_h
			# annot_w =current_an[:,2] - current_an[:,0] 
			# annot_h =current_an[:,3] - current_an[:,1] 
			# select_index = np.where((annot_w>0) & (annot_h>0))
			# current_an = current_an[select_index]
			
			patch_annots.append(current_an)
	if save_img:
		return patch_img, patch_annots 
	else:
		return patch_annots      


pkl_file = '../VisDrone2019-DET-train.pkl'
img_prefix = '../VisDrone2019-DET-train/images'
patch_img_save_file = '../patch/images/'
patch_annot_save_file = '../patch/patch_train.pkl'
patch_h, patch_w = 416, 416
if not os.path.exists(patch_img_save_file):
	os.makedirs(patch_img_save_file)
fr = open(pkl_file, "rb+")
inf = pickle.load(fr)
fr.close()

save_img = True
total_num = len(inf)
total_patch = 0
zero_num = 0
patch_pkl_list = []
for num, annot_dict in enumerate(inf):
	if num % 100 ==0:
		print('processing %d/%d'%(num+1, total_num), end='\r')
	if save_img:
		patch_img, patch_annots = img_patch(img_prefix, annot_dict, patch_w=patch_w, patch_h=patch_h, save_img=save_img)
	else:
		patch_annots = img_patch(img_prefix, annot_dict, patch_w=patch_w, patch_h=patch_h, save_img=save_img)
	total_patch += len(patch_annots)
	
	image_id = annot_dict['filename'].split('.')[0]
	for i, annots in enumerate(patch_annots):
		img_name = image_id + '_%d.jpg'%i
		image_path = os.path.join(patch_img_save_file, img_name)
		if save_img:
			cv2.imwrite(image_path, patch_img[i])
		
		ignore_index = (annots[:,4]==-1)
		labels_ignore = np.zeros(annots[ignore_index,4].shape)
		bboxes_ignore = annots[ignore_index,:4]
		labels = annots[~ignore_index,4]
		bboxes = annots[~ignore_index,:4]
		if bboxes.shape[0] == 0:
			zero_num += 1

		dict_ = {'filename': img_name, 
				'width': patch_w, 
				'height': patch_h, 
				'ann': {'bboxes': bboxes.astype(np.float32),
						'labels': labels.astype(np.int64), 
						'bboxes_ignore': bboxes_ignore.astype(np.float32), 
						'labels_ignore': labels_ignore.astype(np.int64)}
				}
		patch_pkl_list.append(dict_)

output = open(patch_annot_save_file, 'wb')
pickle.dump(patch_pkl_list, output)
output.close()
if total_patch != len(patch_pkl_list):
	print('number error')
print()
print('zero_num: ', zero_num)
print('total_num: ' , total_patch)
			

	

 
