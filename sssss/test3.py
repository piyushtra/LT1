a=[1,2,5]
b=[2,1,6]

# a_temp=[]
# b_temp=[]
# for eachA, eachB in zip(a,b):
#
# 	if eachB%eachA == 0 and eachA !=1:
# 		 k = eachB/eachA
# 		 eachB = eachB/k
# 		 eachA = eachA/k
# 	a_temp.append(eachA)
# 	b_temp.append(eachB)
# a = a_temp
# b = b_temp
print(a)
print(b)
def getLCM(array):
	lcm = 1
	for eachN in array:
	
		if lcm % eachN==0:
			pass
		elif eachN % lcm ==0:
			lcm = eachN
		elif lcm > eachN and lcm%eachN != 0:
			lcm = eachN * lcm
		elif lcm < eachN and eachN%lcm != 0:
			lcm = eachN * lcm
	return lcm
	
#print(getLCM(b))

lcm = getLCM(b)
update_a = []
sumofAs = 0
for eachA, eachB in zip(a,b):

	co = lcm / eachB
	updatedA = eachA * co
	update_a.append(updatedA)
	sumofAs = int(sumofAs + updatedA)


print(str(sumofAs)+"/"+str(lcm))



def getFactors(a):
	primfactor = []
	i = 2
	while a > 1:
		if a%i==0:
			primfactor.append(i)
			a = a/i
		else:
			i = i + 1
	return primfactor
print(getFactors(30))


t = getFactors(sumofAs)
v = getFactors(lcm)
closed_val = []
gcd = 1
for eachValue in set(t):
	if eachValue not in closed_val:
		continue
	xx = t.count(eachValue)
	yy = v.count(eachValue)
	cnt = min(xx,yy)
	closed_val.append(eachValue)
	gcd = gcd * (eachValue**cnt)

print("gcd",str(gcd))
print(str(sumofAs/gcd)+"/"+str(lcm/gcd))
