import json


f = open("airlines.csv", "r")
data = f.read()

json_array = []
splitted_row = data.split("\n")
#print(splitted_row[0])
airport_code_list = []
for i,each_row in enumerate(splitted_row):

    if "," in each_row and i>0:
        airport_code = each_row.split(",")[0]
        time_year = each_row.split(",")[-2]
        statistics_flights_cancelled = each_row.split(",")[-1]
        #print(statistics_flights_cancelled )
        remaining_data = each_row.replace(airport_code,"").replace(time_year,"").replace(statistics_flights_cancelled,"")
        #airport_name = remaining_data.split("\"")[1]
        while remaining_data.startswith(",") or remaining_data.startswith("\""):
            remaining_data = remaining_data[1:]

        while remaining_data.endswith(",") or remaining_data.endswith("\""):
            remaining_data = remaining_data[:-1]

        airport_name = remaining_data
        js = {"airport_code" : airport_code,"airport_name":airport_name,"time_year":time_year,"statistics_flights_cancelled":statistics_flights_cancelled}
        json_array.append(js)

#print(json_array)
#ATL,"Atlanta, GA: Hartsfield-Jackson Atlanta International",2003,216
#BOS,"Boston, MA: Logan International",2003,138

counter_dict = {}
maxcount = 0
maxcount_name =""

mincount = 1000
mincount_name =""
for each_data in json_array:
    airport_name = each_data.get("airport_name")
    if airport_name in counter_dict.keys():
        count = counter_dict.get(airport_name)
        count = count+1
        counter_dict[airport_name] = count

    else:
        counter_dict[airport_name] = 1


for each_counter_dict in counter_dict:
    count = counter_dict.get(each_counter_dict)
    if count > maxcount:
        maxcount = count
        maxcount_name = each_counter_dict

    if count < mincount:
        mincount = count
        mincount_name = each_counter_dict

counter_json = json.dumps(counter_dict, indent = 4)


print(counter_json)
print(maxcount_name,maxcount)
print(mincount_name,mincount)