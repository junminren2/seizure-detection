import os
import time
import torch
import linecache
import shutil

def accu(outfile):
 outopen = open(outfile, 'r', encoding='utf-8')
 #outopen1 = open(outfile1, 'w', encoding='utf-8')

 lines = outopen.readlines()
 #num_file = len(lines)
 j = 0
 fp = 0
 fn = 0
 for line in lines:
    j += 1
    if j % 2 == 0:
     if str(1) in line:
    #if str(1/t1/t1/t1/t1/t1/t1/t1/t1/t/n) in line:

        fp += 1
        print(line)
        #fn += 1


 #acc_1 = (fn / num_file)
 num_file = j/2
 print(j)
 print(num_file)
 print(fp)
 acc_1 = fp / num_file
 print('accuracy = %10.3f' % acc_1)
def accu_1(outfile):
 outopen = open(outfile, 'r', encoding='utf-8')
 #outopen1 = open(outfile1, 'w', encoding='utf-8')

 lines = outopen.readlines()
 #num_file = len(lines)
 j = 0
 fp = 0
 fn = 0
 fp_1 = 0
 for line in lines:
    j += 1
    print(line)
    if j % 2 == 0:
       if str(1) in line:
    #if str(1/t1/t1/t1/t1/t1/t1/t1/t1/t/n) in line:

        fp += 1
       if line.count('1') > 0:
        fp_1 += 1
        # fn += 1

 num_file = j/2
 print(j)
 print(num_file)
 print(fp)
 acc = fp / num_file
 acc_1 = fp_1 / num_file
 print('accuracy = %10.3f' % acc)
 print(fp_1)
 print('accuracy = %10.3f' % acc_1)



outfile = 'E:\\epilepsy_video\\018\\P_2020-06-08_predict.txt'
#fenhang(infile,outfile, outfile1)
#accu(outfile)020_move_predict(1)
accu_1(outfile)

