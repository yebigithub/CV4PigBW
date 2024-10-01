#Author Ye Bi 10/01/2024

import os
import numpy as np
import shutil


rootdir = "YourFolder"
DAY = "T1"
DAY_folder = rootdir + DAY


from ultralytics import YOLO
yolomodel = YOLO("D:\OneDrive_VT\OneDrive - Virginia Tech\Research\Codes\\research\PigBW\Pig_yolo_finetune_Beta\\runs\detect\\train\weights\\best.pt")

def preprocess_yolo(yolomodel, im):
    # YOLOv8 inference
    results = yolomodel(im)
    ress = results[0].boxes  # Accessing the boxes for the first image

    # Convert boxes to the same format as the YOLOv5 pandas dataframe (xyxy, confidence, class, name)
    df = ress.xyxy.cpu().numpy()  # xyxy format
    confidence = ress.conf.cpu().numpy()  # confidence scores
    class_ids = ress.cls.cpu().numpy().astype(int)  # class ids
    class_names = [yolomodel.names[class_id] for class_id in class_ids]  # get class names

    # Create a dataframe-like structure for easy condition checking
    ress_df = {
        'xmin': df[:, 0], 'ymin': df[:, 1], 'xmax': df[:, 2], 'ymax': df[:, 3],
        'confidence': confidence, 'name': class_names
    }

    condition1 = len(ress_df['name']) <= 0  # no detection, skip
    condition2 = sum(confidence[i] < 0.5 for i, name in enumerate(ress_df['name']) if name == 'pig')  # pig less than 0.5, skip
    condition3 = sum(confidence[i] > 0 for i, name in enumerate(ress_df['name']) if name == 'block')  # block detected, skip

    if condition1 or condition2 > 0 or condition3 > 0:
        return False
    else:
        return True
    
    
for pen in os.listdir(DAY_folder):
  if not pen.endswith('.sh'):
    print("Now is running pen number", pen)
    dep_folder = rootdir + DAY + "/" + pen + "/Depth/"
    for bag_id in os.listdir(dep_folder):
      #   print("Now is running bad id", bag_id)
        depthdir = dep_folder + bag_id + "/"
        depthdir_after = rootdir + DAY + "/" + pen + "/Depth_afterYOLO/" + bag_id + "/" #define the new folder
        # if os.path.exists(depthdir_after):
        #   print("Already exist, skip this bag id")
        #   continue

        for root, dirs, files in os.walk(depthdir):
              start = round(len(files)*0)
              end = round(len(files)*1)
              
              for j in np.arange(start, end, 1): # one pic per 15 frames
                  file = files[j]
                  file_path = os.path.join(root, file)
                  # print("Now is running: ", file_path)

                  ##YOLO detection
                  yolo_detection_QC = preprocess_yolo(yolomodel, im = file_path)
                  if yolo_detection_QC == False:
                      continue
                  else:
                      os.makedirs(depthdir_after, exist_ok=True) #make new folder named "Depth1" ---- to keep new images.
                      shutil.copy(file_path, depthdir_after+file) #copy good images into Depth1, remove trash images.
                                          
                 



