
# Check if the module is installed and the file path is correct
from cgi import test
import re
from Dueling_GANDDQN import action_space
import numpy as np
class Testfunc():
    def __init__(self):
        self
    def testnp_where(self):
        result = [[0,0,1],[0,1,0],[0,0,1]]
        result=np.array(result)
        [i,j]=np.where(result == 0)
        print(i,j) 
    def testpriorityofPowAndMul(self):
        a=160 * 10 ** (-3)
        print(a)
    # 数组乘以数字不会改变数组长度，但是列表乘数字却会让列表长度变长
    def testListMul(self):
        a=[0]*3
        print(a)
    def testmatrixcompare(self):
        UE_index = np.array(8,14)
        ser_schedu_ind = [0]*3
        left_no = np.where(UE_index > self.ser_schedu_ind[1])[0].size  #这是个什么含义 
    def testCompare(self):
         # numpy的比较运算

        # 声明2个numpy 数组
        arr1 = np.array([11, 12, 13])
        arr2 = np.array([21, 22, 23])

        # numpy的2种比较运算方法， 第一种是使用运算符
        '''
        大于 >
        小于 <
        小于等于 <=   大于等于 >=
        比较值是否相等  ==
        不等于  ！=
        '''

        print('arr1大于arr2吗？ {}'.format(arr1 > arr2))
        print('arr1小于arr2吗？ {}'.format(arr1 < arr2))
        print('arr1大于等于arr2吗？ {}'.format(arr1 >= arr2))
        print('arr1小于等于arr2吗？ {}'.format(arr1 <= arr2))
        print('arr1和arr2一样吗？ {}'.format(arr1 == arr2))
        print('arr1不等于arr2吗？ {}'.format(arr1 != arr2))

        # 第二种是使用方法去做比较
        '''
        大于 np.greater()
        小于 np.less()
        大于等于  np.greater_equal()
        小于等于    np.less_equal()
        比较值是否相等 np.equal()
        不等于  np.not_equal()
        '''

        print('arr1大于arr2吗？ {}'.format(np.greater(arr1, arr2)))
        print('arr1小于arr2吗？ {}'.format(np.less(arr1, arr2)))
        print('arr1大于等于arr2吗？ {}'.format(np.greater_equal(arr1, arr2)))
        print('arr1小于等于arr2吗？ {}'.format(np.less_equal(arr1, arr2)))
        print('arr1等于arr2吗？ {}'.format(np.equal(arr1, arr2)))
        print('arr1不等于arr2吗？ {}'.format(np.not_equal(arr1, arr2)))
    def testmatmul(self):
        a=np.array([[1,2,3],[4,5,6]])
        b=np.array([[1,2],[3,4],[5,6]])
        print(np.matmul(a,b))
    def testFile(self):
        output = np.load('/data/GANDDQN/result_10M_1M_LURLLC_2024-02-18.npz')
        print(output)
if __name__ == '__main__': 
    test=Testfunc()
    test.testmatmul()