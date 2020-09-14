import os
import time
import torch
import linecache
import shutil

def fenhang(infile, output):

    infopen = open(infile,'r',encoding='utf-8')
    #outopen = open(outfile,'w',encoding='utf-8')
    #outopen1 = open(outfile1, 'w', encoding='utf-8')
    lines = infopen.readlines()
    i = 0
    fp = 0
    for line in lines:
        i += 1
        if i % 2 == 0:
            if str(1) in line:
                fp += 1
                print(linecache.getline(infile, i).strip())
                shutil.move(linecache.getline(infile, i-1).strip(), output)
    print(fp)
    infopen.close()
def accu(input, outfile,outfile1, output):
 outopen = open(outfile, 'r', encoding='utf-8')
 #outopen1 = open(outfile1, 'w', encoding='utf-8')
 file_list = os.listdir(input)
 num_file = len(file_list)
 print(num_file)
 lines = outopen.readlines()
 j = 0
 fp = 0
 fn = 0
 for line in lines:
    j += 1
    if str(1) in line:
    #if str(1/t1/t1/t1/t1/t1/t1/t1/t1/t/n) in line:
        fp += 1
        #fn += 1
        print(linecache.getline(outfile1, j).strip())
        shutil.move(linecache.getline(outfile1, j).strip(), output)


 #acc_1 = (fn / num_file)
 acc_1 = fp / num_file
 print('accuracy = %10.3f' % acc_1)


infile = 'D:\\test_026\\finish.txt'
pred = infile
input = 'D:\\test_YJ-022-026\\P_YJ_026_MF_20200616'
#outfile = 'D:\\test_YJ-022-026\\' + 'P_control_YJ-026_predict_model_mice_5_1.txt'
#outfile1 = 'D:\\test_YJ-022-026\\' + 'P_control_YJ-026_predict_model_mice_5_2.txt'
output = 'D:\\test_026\\pre_026_fn'
fenhang(infile, output)
#accu(input, outfile, outfile1, output)

