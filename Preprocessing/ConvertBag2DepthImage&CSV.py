#Author Ye Bi 10/01/2024
```
Don't forget to put BagConvert.py(https://github.com/yebigithub/CV4PigBW/blob/main/Preprocessing/BagConvert.py) 
inside of the same folder with this code.
Also you need to download Intel RealSense SDK 2.0 (https://www.intelrealsense.com/sdk-2/)
```

from BagConvert import ConvertToPNG as convert

import os
con = convert()
DayFolder = "YourBagFolder"
DayOutput = "YourOutputFolder"
toolsPath = "C:\Program Files (x86)\Intel RealSense SDK 2.0\\tools"
InputFolders =  os.listdir(DayFolder)

for InputFolder in os.listdir(DayFolder):
        if os.path.isdir(os.path.join(DayFolder, InputFolder)):
            print(InputFolder)
            outputFolder_depth = DayOutput+InputFolder+"\Depth\\"
            outputFolder_csv = DayOutput + InputFolder + "\CSV\\"

            inputFolder = os.path.join(DayFolder, InputFolder)
            con.convert(inputFolder,
                        outputFolder_depth,
                        outputFolder_csv,
                        toolsPath)