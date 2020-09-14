import numpy as np
import os
import shutil
import xlrd
import cv2
import config
import random
import time
def move_case(excel_path,newdir, goal_path1):
    excel=xlrd.open_workbook(excel_path)
    table=excel.sheet_by_index(0)
    rows=table.nrows
    #表格第一列为视频名名，第二列为开始时间，第三列结束时间，第四列类型 1 case 2 control
    ori_video=table.col_values(0, start_rowx=0, end_rowx=None)

    for i in range(rows):
            path=os.path.join(newdir, ori_video[i])
            shutil.move(path,goal_path1)
excel_path = 'E:\\epilepsy_video\\move.xlsx'
newdir = 'E:\\epilepsy_video\\test_cut_P_case_YJ_John_L04\\cut_P_case_YJ_John_L04'
goal_path1 = 'E:\\epilepsy_video\\test_cut_P_case_YJ_John_L04\\cut_P_case_YJ_John_L04\\no_detect'
move_case(excel_path,newdir, goal_path1)