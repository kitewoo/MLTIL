import numpy as np

list1 = [1,2,'str']
print(np.array(list1)) #int & str -> str 

list2 = [1,2, 3.1]

array2 = np.array(list2)

print(np.array(list2)) #int & float -> float (float data is bigger than int)

#but we can change the dtype as astype('')

print(array2.astype('int32'))

#dimension

# 1차원 
d1 = np.array([1,2,3])
print(d1.shape)
# 2차원
d2 = np.array([[1,2,3]])
print(d2.shape)
# 3차원
d3 = np.array([[[1,2,3]]])
print(d3.shape)


# 편한 생성 arange, zeors, ones

sequence_array = np.arange(10)
print(sequence_array)

zero_array = np.zeros((3,2), dtype = 'int32')
print(zero_array)

one_array = np.ones((3,2), dtype = 'int32')
print(one_array)


# reshape : change dimension

array1 = np.arange(10)
print('array1: \n', array1)

array2 = array1.reshape(-1,5) #-1
print('array2:\n', array2)

array3 = array1.reshape(2,5)
print('array3:\n', array3)

# indexing
print(array1[2])

array1[9] = 11 #replace
print(array1[5:-1]) #slicing

# 2차원 indexing
print(array2[0:2,0:3])
print(array2[[0,1], 0:3]) #팬시 인덱싱 방법 : 행, 열로 인덱싱
# boolean indexing
print(array1 > 5) 

