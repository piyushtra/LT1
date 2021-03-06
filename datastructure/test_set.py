setA = set([1,2,3,4,5,1,2,3])
print(setA)

setB = {"piyush","dev","rajeev","aditya",3}
print(setB)

setB.add("malini")
print(setB)

setB.discard("malini")
print(setB)

for each in setB:
    print(each)


set_u = setA|setB
print(set_u)

set_i = setA & setB
print(set_i)

set_sub = setA - setB
print(set_sub)

compare = setA > setB
print(compare)

setC = {1,2,3,4,5,6,7,8,9,0}
print(setC > setA)
print(setC < setA)
print(setC >= setA)

