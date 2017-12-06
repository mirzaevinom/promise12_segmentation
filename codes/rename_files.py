import os
import shutil

fileList = filter(lambda x: 'qsub' in x, os.listdir('./') )

fileList.sort()
shutil.move(fileList[-1], 'output.txt')
