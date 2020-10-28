import glob
txts = glob.glob('/home/junminren/Mice_ictal_recognition-master/detection/coco128/labels/label_1/*.txt')
for one_txt in txts:
    print(one_txt)
    f = open(one_txt, 'r+', encoding='utf-8')
    all_the_lines = f.readlines()
    f.seek(0)
    f.truncate()
    for line in all_the_lines:
       line = line.replace('15 ', '0 ')
       f.write(line)
    f.close()