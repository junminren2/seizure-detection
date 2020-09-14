import os
import imutils
import shutil
def move_all_file(dir, out_dir):
 path = os.path.join(dir)
 subdirs = os.listdir(path)
 for subdir in subdirs:
    # compute the path to the subdir
    path = os.path.join(dir, subdir)
    elems = os.listdir(path)
    for elem in elems:
        # is it a directory?  If so, process it...
        # get all of the elements in the subdirectory
        subpath = os.path.join(path, elem)

        shutil.copy(subpath, out_dir)


dir = 'E:\\data_YU_20200606\\09_09_control_dataset'
out_dir = 'E:\\data_YU_20200606\\09_09_control_dataset\\out'
if not os.path.exists(out_dir):
   os.makedirs(out_dir)

move_all_file(dir,out_dir)