import numpy as np
from multiprocessing import Pool
from PIL import Image
from Feature_extraction import *
import threading
import gc
import os
from datetime import datetime
import linecache

def feature_extract(dir, index):
    print(dir,index)
    im = Image.open(dir).convert('L')
    im = im.resize((24, 24))
    #im.show()
    im_array = np.array(im)
    #print('start to extract')
    img_feature = NPDFeature(im_array).extract()

    list_dir = dir.split('\\')
    list_dir.pop()
    list_dir.pop()
    dir = '\\'.join(list_dir)
    data_dir = dir +'\\data.txt'
    position_dir = dir + '\\position.txt'
    pos = linecache.getline(data_dir, index)
    position = linecache.getline(position_dir, index)
    #print(pos)
    #print(position)

    pos_array = np.fromstring(pos, dtype= float, sep = ' ')                    #字符串变到numpy array
    position_array = np.fromstring(position, dtype = int, sep = ' ')

    img_feature = np.append(img_feature,position_array )                  #在NPD特征张量末尾加上目标物体位置以及pose真值
    img_feature = np.append(img_feature, pos_array)

    #print('img '+ dir+ ' finish')
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

def connect_feature(pos_result_list):
    features_array = np.array(pos_result_list[0])
    triple_list = []
    feature_block = []
    thread_list = []
    image_num = len(pos_result_list)  # 185
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
    return features_array




if __name__ == '__main__':
    print('begin at ' + datetime.now().strftime('%H:%M:%S'))

    #pos_dir = 'L:\\Dataset\\images\\7.3\\1530598544_pos\\image\\'
    #neg_dir = 'L:\\Dataset\\images\\7.3\\1530598544_pos\\image\\'
    pack_dir = 'L:\\Dataset\\images\\7.3\\train\\'
    #pos_dir =  'L:\\Dataset\\images\\7.3\\train\\1530598544_pos\\'
    #neg_dir = 'L:\\Dataset\\images\\7.3\\train\\1530600404_pos\\'

    block_list = []
    dirs = os.listdir(pack_dir)

    for pos_dir in dirs:
        pos_img_dir = []
        pos_img_list = []
        #print(pos_dir)
        for root, dirs, files in os.walk(pack_dir + pos_dir + '\\image\\', topdown=False):
            i = 1
            for file in files:
                pos_img_dir.append(pack_dir + pos_dir + '\\image\\' + file)
                pos_img_list.append(i)
                i += 1
        #print('size:', pos_img_dir, pos_img_list)

        tasks = zip(pos_img_dir, pos_img_list)
        #print (tasks)
        multi_feature_extraction_pool = Pool(processes = 4)
        pos_result_list = multi_feature_extraction_pool.starmap(feature_extract, tasks)
        pos_result = connect_feature(pos_result_list)
        block_list.append(pos_result)
        print('finish pos_feature_extraction')


    print('block_list',len(block_list))
    feature_array = block_list[0]

    for i in range(1,len(block_list)):
        feature_array = np.row_stack((feature_array, block_list[i]))

    np.save(pack_dir + 'feature_data_array', feature_array)

    print('feature save complete')
