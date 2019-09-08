from imutils import face_utils
import numpy as np
import imutils
import os
import dlib
import cv2

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat') 

# extract lips

for root, dirs, files in os.walk("dataset", topdown=False):
   for name in files:
        if "color" in name and name.endswith(".jpg") and not "lips" in root: # only RGB images
            fname = os.path.join(root,name)
            lips_dir = os.path.join(root,"lips")
            if os.path.exists(lips_dir) == False:
                os.mkdir(lips_dir)
            print(lips_dir)
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 1)
            for (i,rect) in enumerate(rects):
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                for (n,(i,j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
                    if n=='mouth':
                        for (x, y) in shape[i:j]:
                            (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
                            roi = img[y:y + h, x:x + w]
                        lname = os.path.join(lips_dir,'lips_'+name)
                        cv2.imwrite(lname,roi)

# join images 

for root, dirs, files in os.walk("dataset", topdown=False):
   for name in dirs:
        if "lips" in name: # only RGB images
            fdir = os.path.join(root,name)
            print(fdir)
            # create concatenated image
            seq = np.zeros((224,224))
            orig = len(os.listdir(fdir))
            orig_image = []
            for i in os.listdir(fdir):
                i = os.path.join(fdir,i)
                img = cv2.imread(i,0)
                img = cv2.resize(img,(32,32),interpolation = cv2.INTER_LINEAR)
                orig_image.append(img)
            x_limit = 32
            y_limit = 32
            iterator_x = 0
            iterator_y = 0
            for i in range(0,49):
                seq[iterator_y:iterator_y+y_limit,iterator_x:iterator_x+x_limit] = orig_image[int((i*orig)/49)] # movement along y-axis
                iterator_x = iterator_x + x_limit
                if iterator_x == 224:
                    iterator_x = 0 
                    iterator_y = iterator_y + y_limit 
            fdir_split = fdir.split("/")
            name = "dataset/" + fdir_split[1] + "_" + fdir_split[2] + fdir_split[3] + "_" + fdir_split[4] +'.jpg'
            cv2.imwrite(name,seq)



### CREATE folder structure for ImageFolderGenerator Keras
from shutil import copyfile

### CREATE folder structure for ImageFolderGenerator Keras
strn = 'F01_phrases01_01.jpg'

fn,ext = os.path.splitext(strn) 
part1,part2,part3 = fn.split('_')

idx = part2[-2:]
i = 0 
if idx[0]==0:
    i = int(idx[1]) - 1
else:
    i = int(idx) - 1
if part2[:-2] == 'phrases':
    i = i
else:
    i = i + 10

for f in os.listdir("dataset"):
    f_with_path = os.path.join("dataset",f)
    if os.path.isdir(f_with_path) == False:
        fn,ext = os.path.splitext(f_with_path) 
        part1,part2,part3 = fn.split('_')
        idx = part2[-2:]
        i = 0 
        if idx[0]==0:
            i = int(idx[1]) - 1
        else:
            i = int(idx) - 1
        if part2[:-2] != 'phrases':
            i = i + 10
        new_dir = os.path.join("dataset",str(i))
        if os.path.exists(new_dir) == False:
            os.mkdir(new_dir)
        name = new_dir + "/" + f
        print(name)
        copyfile(f_with_path,name)
