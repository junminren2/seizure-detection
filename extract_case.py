import os
import numpy as np
import shutil
import xlrd
import configparser
import functools


# Function to convert
# def listToString(s):
#     # initialize an empty string
#     str1 = " "
#
#     # return string
#     return (str1.join(s))
def extra_case(exp, num, excel_path, new_dir):
    """
        Function used to cut the video according to the excel information
        :param excel_path: string, the path of the excel file
        :param new_dir: string, the path to storage the result
        :param goal_paths: dictionary, contains path of different classes
    """
    excel = xlrd.open_workbook(excel_path)
    table = excel.sheet_by_index(0)
    rows = table.nrows
    #First row is the video fiel name
    d = table.col_values(0, start_rowx=0, end_rowx=None)

    h = table.col_values(1, start_rowx=0, end_rowx=None)
    h = list(map(int, h))
    m = table.col_values(2, start_rowx=0, end_rowx=None)
    m = list(map(int, m))






    for i in range(rows):
        # d = listToString(d)
        # h = listToString(h)
        # m = listToString(m)

        # filename, file_format = raw_video[i].split('.')
        # date = filename[7:-6]
        # print(date)
        # elems = os.listdir(new_dir)
        # d = ''.join(str(a) for a in d[i])
        # h = ''.join(str(int(b)) for b in h[i])
        # m = ''.join(str(int(c)) for c in m[i])
        if len(str(h[i])) == 1:
            h0 = '0' + str(h[i])
        else:
            h0 = str(h[i])
        if len(str(m[i])) == 1:
            m0 = '0' + str(m[i])
        else:
            m0 = str(m[i])

        dir_path = os.path.join(new_dir, str(d[i]))
        move_dir = os.path.join(dir_path, h0)
        file_path = os.path.join(move_dir, exp + '_' + num + '_' + d[i] + '_' + h0 + '_' + m0 + '.mp4')
        goal_paths = os.path.join(new_dir, 'extract')
        if not os.path.exists(goal_paths):
            os.makedirs(goal_paths)
        shutil.copy(file_path, goal_paths)





if __name__ == '__main__':
    exp = 'YJ'
    num = '012'
    excel_path = 'D:\\videoAI\\cut.xlsx'
    new_dir = 'D:\\videoAI\\2020-06-18\\move'
    # goal_paths = 'D:\\videoAI\\'
    extra_case(exp, num, excel_path, new_dir)


