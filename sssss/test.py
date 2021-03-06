'''
def minion_game(string):
    # your code goes here
    temp = string
    vc = 0
    cc = 0

    for eachCh in range(65, 91):
        ch = chr(eachCh)
        if ch != "A" and ch != "E" and ch != "I" and ch != "O" and ch != "U":
            cc = cc + getPC(string, ch)
        else:
            vc = vc + getPC(string, ch)

    if cc > vc:
        print("Stuart " + str(cc))
    else:
        print("Kevin " + str(vc))


def getPC(string, ch):
    pc = 0
    temp = string
    if ch in temp:
        i = temp.find(ch)
        pc = len(temp) - i
        temp = temp[i + 1:]
        if ch in temp:
            pc = pc + getPC(temp, ch)
    return pc


if __name__ == '__main__':
    s = input()
    minion_game(s)

'''
'''
##########
def subset_sum(numbers, target, partial=[]):
    s = sum(partial)

    # check if the partial sum is equals to target
    if s == target: 
        print("sum(%s)=%s" % (partial, target))
    if s >= target:
        return  # if we reach the number why bother to continue

    for i in range(len(numbers)):
        n = numbers[i]
        remaining = numbers[i+1:]
        subset_sum(remaining, target, partial + [n]) 


if __name__ == "__main__":
    subset_sum([3,9,8,4,5,7,10],15)
#######
'''

strt="aebcbda"
temp = strt
l1 = len(strt)

ss = strt.split()
alls = []
lenalls = []
for startIndex in range(len(strt)):
    index = startIndex
    strt = temp
    ini=""
    l=""
    m=""
    while strt != "" and index<len(strt):
        each = strt[index]
        i = strt.rfind(each)
        if index == i :
            m = each
            strt=strt[1:]
        else:
            ini = ini + each
            strt = strt[1:i]
            l = each + l
    alls.append(str(ini)+str(m)+str(l))
    lenalls.append(len(str(ini)+str(m)+str(l)))

print(alls)
print(str(ini)+str(m)+str(l))
l2 = max(lenalls)
print(l1 - l2)