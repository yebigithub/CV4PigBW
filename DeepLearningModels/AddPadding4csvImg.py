#Author Ye Bi 10/01/2024

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import os
import numpy as np
import pandas as pd
import tensorflow as tf





import tensorflow as tf
print("GPU is using now:", tf.config.list_physical_devices('GPU'))

os.chdir("/home/yebi/ComputerVision_PLF/Pig_BW/Pig_BW_DL_beta/DL/ResNet")
# df1 = pd.read_csv("../labelled_depth_0718.csv")
# df2 = pd.read_csv("../labelled_depth_0801.csv")
# df3 = pd.read_csv("../labelled_depth_0815.csv")
df4 = pd.read_csv("../labelled_depth_0912.csv")
df5 = pd.read_csv("../labelled_depth_0927.csv")
df6 = pd.read_csv("../labelled_depth_0829.csv")

visit = "all"
# # labelled_depth = pd.concat([df1, df2, df3], axis=0)
# if visit == "0718":
#     labelled_depth = df1
# elif visit == "0801":
#     labelled_depth = df2
# elif visit == "0815":
#     labelled_depth = df3
if visit == "0912":
    labelled_depth = df4
elif visit == "0927":
    labelled_depth = df5
elif visit == "0829":
    labelled_depth = df6
elif visit == "all":
    labelled_depth = pd.concat([df4, df5, df6], axis=0)


def read_images(labelled_depth):
    # images = []
    # img_paths = []
    for filename in labelled_depth["FilePath"]:
        print(filename)
        if filename.endswith('.png'):  
            if os.path.exists(filename):
                img_path = filename
                img = tf.io.read_file(img_path)
                img = tf.image.decode_png(img, channels=3)  
                img = tf.image.resize_with_crop_or_pad(img, target_height=int(img.shape[1]), target_width=int(img.shape[1]))
                img_encoded = tf.image.encode_png(img)
                # images.append(img)
                # img_paths.append(img_path)
                tf.io.write_file(img_path, img_encoded)

    print("DDDDDDDDDDDDDDDDDDDDDDDD")
######################################################################

print("Now is reading training sets")
read_images(labelled_depth=labelled_depth)


