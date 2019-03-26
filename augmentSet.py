# Augmenting the dataset
from tqdm import tqdm
import numpy as np
import os
import cv2

def augment(training_list, labels_list):

    cwd = os.getcwd()

    newDir = os.path.join(cwd, "dataset/train/fold_3")
    allDir = os.path.join(newDir, "all")
    hemDir = os.path.join(newDir, "hem")

    if not (os.path.isdir(newDir)):
        os.makedirs(newDir)
    else:
        print("fold_3 already exists")
    if not (os.path.isdir(allDir)):
        os.makedirs(allDir)
    else:
        print("all folder already exists")
    if not (os.path.isdir(hemDir)):
        os.makedirs(hemDir)
    else:
        print("hem folder already exists")

    for img in training_list:

        originalImg = cv2.imread(img)

        if (img.split('/')[1] != 'dataset'):
            # print(img.split('/')[1])
            del img
            continue

        # rotate the image
        num_rows, num_cols = originalImg.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 30, 1)
        rotated = cv2.warpAffine(originalImg, rotation_matrix, (num_cols, num_rows))

        name = 'rotated_'+img.split('/')[5]
        category = name.split('_')[5]

        category = category.split('.')[0]
        # print (category)
        if category == "hem":
            cv2.imwrite(hemDir+'/'+name, rotated)
            training_list.append(os.path.join(hemDir, name))
            labels_list.append(0)

        else:
            cv2.imwrite(allDir+'/'+name, rotated)
            training_list.append(os.path.join(allDir, name))
            labels_list.append(1)


    return training_list, labels_list
