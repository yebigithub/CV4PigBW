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