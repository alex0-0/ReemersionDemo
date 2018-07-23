#!/usr/bin/env python3
""" Utility functions """
import shutil, random, os

TAG = "UTILITY\t"

'''
-----------------------------------------------------------------------------------
| Function          |Copies Metadata|Copies Permissions|Can Use Buffer|Dest Dir OK
-----------------------------------------------------------------------------------
| shutil.copy       |      No       |        Yes       |      No      |    Yes
-----------------------------------------------------------------------------------
| shutil.copyfile   |      No       |        No        |      No      |    No
-----------------------------------------------------------------------------------
| shutil.copy2      |      Yes      |        Yes       |      No      |    Yes
-----------------------------------------------------------------------------------
| shutil.copyfileobj|      No       |        No        |      Yes     |    No
-----------------------------------------------------------------------------------
'''
"copy random n files from directory d to directory t"
def randomlyCopyFiles(d, t, n):
    if os.path.isdir(d) == False or os.path.isdir(t) == False:
        print(TAG + "source or destination is not a directory")
        return 
    file_num = len([name for name in os.listdir(d) if os.path.isfile(os.path.join(d, name))])
    #the number of files in directory d is less than the number of files we want to pick
    if file_num < n:
        print(TAG + "the number of files in directory" + d + "is less than the number of files you want to pick")
        return 

    filenames = random.sample(os.listdir(d), n)
    for f in filenames:
        srcpath = os.path.join(d, f)
        print(srcpath)
        shutil.copy2(srcpath, t)
'''     BASH
ls |sort -R |tail -$N |while read file; do
    # Something involving $file, or you can leave
    # off the while to just get the filenames
done
'''

#randomlyCopyFiles("/home/alex/Documents/work/train2017", "images", 200)
