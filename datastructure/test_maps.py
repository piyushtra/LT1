import collections

dict1 = {"day1": "Mon","day2":"Tue"}
dict2 = {"day3": "Wed", "day1": "Thu"}

res = collections.ChainMap(dict1,dict2)
print(res)

print(res.maps)

print("Keys =  {}". format(list(res.keys())))
print("Values =  {}".format(list(res.values())))

print("elements :")
for key,value in res.items():
    print("{} = {} ".format(key,value))


print("day3 in res : {}".format("day3" in res))
print("day4 in res : {}".format("day4" in res))


dict1["day4"]="Thu"
print(res.maps)
