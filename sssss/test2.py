# Python3 Program to get the maximum
# possible integer from given array
# of integers...


# custom comparator to sort according
# to the ab, ba as mentioned in description
from functools import cmp_to_key

def comparator(a, b):
	ab = str(a) + str(b)
	ba = str(b) + str(a)
	if ab >= ba:
		return -1
	else:
		return 1




# driver code
if __name__ == "__main__":
	#a = [54, 546, 548, 60, ]
	a = [1,34,3,98,9,76,45,4]
	sorted_array = sorted(a, key=cmp_to_key(comparator))
	number = "".join([str(i) for i in sorted_array])
	print(number)

# This code is Contributed by SaurabhTewary

