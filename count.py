str='hello'
same=''
diff=''
if str.count('o') > 1:
        same+='1'
else:
        diff+='1'
print('重复的元素有：%s'%same)
print('不重复的元素有：%s'%diff)