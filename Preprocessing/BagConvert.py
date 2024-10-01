#Author Ye Bi 10/01/2024
#this script uses rs-convert to convert .bag files inot PNG format

import os

class ConvertToPNG:

    def __init__(self):
        print('Starting Conversion')
        print('hello')

    def convert(self, inputFolder, outputFolder_depth, outputFolder_csv, toolsPath):
        #navigate to RealSense Tools
        os.chdir(toolsPath) ##PATH TO \\Intel RealSense SDK 2.0\\tools\

        commandIntro = 'cmd /c '
        commandPNG = 'rs-convert -d -i '
        #iterate over files in directory

        for filename in os.listdir(inputFolder):
            if filename.endswith('.bag'):
                f = os.path.join(inputFolder, filename)
                print(filename)
                new_name = filename.split(".")[0]

                if os.path.exists(outputFolder_depth + new_name):
                    print(outputFolder_depth + new_name, "exit, go to next file \n")
                    continue

                os.system("mkdir "+ outputFolder_depth + new_name + "\\")
                os.system("mkdir "+ outputFolder_csv + new_name + "\\")
                fullCommandPNG = commandIntro + commandPNG + f + ' '  + '-p ' + outputFolder_depth + new_name + "\\"
                fullCommandCSV = commandIntro + commandPNG + f + ' ' + '-v ' + outputFolder_csv + new_name + "\\"
                
                #convert to PNG-Depth
                print(fullCommandPNG)
                os.system(fullCommandPNG)
                # #convert to CSV
                print(fullCommandCSV)
                os.system(fullCommandCSV)