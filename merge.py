import os
def emerge_all(newdir, old_str, new_str, old_str_1, new_str_1, new_name):
  video_list = os.listdir(newdir)
  video_list.sort()
  f = open(newdir + 'file.txt', 'a')
  for file in video_list:
   line = os.path.join(newdir, file)
   print(line)
   if old_str in line:
    line = line.replace(old_str, new_str)
   if old_str_1 in line:
    line = line.replace(old_str_1, new_str_1)
    f.write(line)
    print(line)
    f.write('\n')
  f.close()
  os.system('ffmpeg -f concat -safe 0 -i %s -codec copy %s' % (newdir + 'file.txt', newdir + new_name))
newdir = 'C:\\Users\\User\\Videos\\2020-03-24\\2020-03-24\\00\\'
old_str = '\\'
new_str = r'\\'
old_str_1 = newdir[:2]
new_str_1 = 'file ' + old_str
new_name = 'output.mp4'
emerge_all(newdir, old_str, new_str, old_str_1, new_str_1, new_name)
#os.system('ffmpeg -y -i %s -ss %i -t %i -codec copy %s' % (path,new_name))