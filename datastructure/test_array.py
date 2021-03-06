from array import *
array1 = array('i',[10,20,30,40,50,60])
for i in array1:
    print(i)
    
    
    
print(array1[1])


array1.insert(1,60)

print(array1)


array1.remove(20)
print(array1)

array1.insert(1,20)
array1.insert(2,20)
print(array1)

array1.remove(20)
print(array1)


print(array1.index(50))


array1[0]=100

print(array1)
