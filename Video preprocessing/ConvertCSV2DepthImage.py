import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import tensorflow as tf

parser = argparse.ArgumentParser(description = 'Convert depth csv files into depth images')
parser.add_argument('--root', dest="root", help = 'root directory info.')
parser.add_argument('--day', dest='day', help = 'day info.')
args = parser.parse_args()

rootdir = args.root
DAY = args.day
DAY_folder = rootdir + DAY

for pen in os.listdir(DAY_folder):
  if pen.endswith("top"):
    print("Now is running pen number", pen)
    dep_folder = rootdir + DAY + "/" + pen + "/CSV/"
    for bag_id in os.listdir(dep_folder):
        print("Now is runng bag_id: ", bag_id)
        depthdir = dep_folder + bag_id + "/"
        depthdir_after = rootdir + DAY + "/" + pen + "/CSV_afterYOLO/" + bag_id + "/"  #After Quality Control, save CSV files into CSV_afterYOLO
        yolodir = rootdir + DAY + "/" + pen + "/Depth_afterYOLO/" + bag_id + "/" #This is the depth images who passed QC.
        for root, dirs, files in os.walk(depthdir):

          start = round(len(files)*0)
          end = round(len(files)*1)
          
          for j in np.arange(start, end, 1): 
              file = files[j]
              frame_id = file.split(".csv")[0]

              output_path = depthdir_after + frame_id + ".png"

              if not os.path.exists(output_path):
                # print(frame_id)
                if os.path.exists(yolodir+frame_id+".png"):
                  # print("Passing yolo checking")
                  file_path = os.path.join(root, file)
                  # print("Now is running: ", file_path)

                  # Read the CSV file into a DataFrame
                  dfcsv = pd.read_csv(file_path, header=None)

                  # Convert DataFrame to NumPy array
                  dfcsv_array = dfcsv.iloc[1:, :].values

                  # Get the shape of the array
                  h0, w0 = dfcsv_array.shape

                  os.makedirs(depthdir_after, exist_ok=True) 

                  plt.figure(figsize=(w0, h0), dpi=0.5)  
                  plt.imshow(dfcsv_array, cmap='gray', vmin=0, vmax=1)
                  plt.axis("off")
                  output_path = depthdir_after + frame_id + ".png"
                  plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
                  plt.close()
              
                                          
# To runthis code
# start cmd /c python ConvertCSV2DepthImage.py --root "YourFolder" --day T1                