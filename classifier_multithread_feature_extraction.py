import numpy as np
from multiprocessing import Pool
import cv2
from Feature_extraction import *
import threading
import gc
import os
from datetime import datetime

def feature_extract(dir):
    im = cv2.imread(dir,0)
    im = cv2.resize(im,(24, 24))
    im_array = np.array(im)
    img_feature = NPDFeature(im_array).extract()
    print('img '+ dir.split('\\')[1]+ ' finish')
    return img_feature

def feature_connect(triple_list, feature_block):
    pos_result_list = triple_list[0]
    i = triple_list[1]
    j = triple_list[2]
    #print(i,j)
    tmp_list = pos_result_list[i:j]
    #print('size', len(tmp_list))
    features_array = np.array(tmp_list[0])
    for n in range(1,len(tmp_list)) :
        features_array = np.row_stack((features_array, tmp_list[n]))
    feature_block.append(features_array)
    print('block',i,'joint complete')

def connect_feature_save(pos_result_list, name):
    features_array = np.array(pos_result_list[0])
    triple_list = []
    feature_block = []
    thread_list = []
    image_num = len(pos_result_list)  # 185
    print(name,' img: ',image_num)
    i = 1
    while i + 100 < image_num:
        triple_list.append([pos_result_list, i, i + 100])
        i += 100
    triple_list.append([pos_result_list, i, image_num])  # 1,101,201,256         [(1,101),(101,201),(201,256)]             #split block

    for item in triple_list:
        t = threading.Thread(target = feature_connect, args = (item,feature_block,))                                     #calculate block
        t.setDaemon(True)
        thread_list.append(t)
    for t in thread_list:
        t.start()
    for t in thread_list:
        t.join()

    for block in feature_block:
        features_array = np.row_stack((features_array, block))                                          #join all blocks

    np.save(name + '_feature_data_array', features_array)
    print(name,'feature save complete')


if __name__ == '__main__':
    print('begin at ' + datetime.now().strftime('%H:%M:%S'))
    data_dir = 'F:\\D2CO_dataset\\detect_train_data\\test\\'
    pos_dir = data_dir + 'pos\\'
    neg_dir = data_dir + 'neg\\'
    pos_img_dir = []
    neg_img_dir = []


    for root, dirs, files in os.walk(pos_dir, topdown=False):
        for file in files:
            pos_img_dir.append(pos_dir+file)

    for root, dirs, files in os.walk(neg_dir, topdown=False):
        for file in files:
            neg_img_dir.append(neg_dir+file)


    multi_feature_extraction_pool = Pool(processes = 8)

    pos_result_list = multi_feature_extraction_pool.map(feature_extract, pos_img_dir)
    print('finish pos_feature_extraction')

    print('start to joint pos feature and save ')
    connect_feature_save(pos_result_list, data_dir+'pos')

    del pos_result_list
    gc.collect()

    neg_result_list = multi_feature_extraction_pool.map(feature_extract, neg_img_dir)
    print('finish neg_feature_extraction')

    print('start to joint neg feature and save ')
    connect_feature_save(neg_result_list, data_dir + 'neg')

    print('end at ' + datetime.now().strftime('%H:%M:%S'))






