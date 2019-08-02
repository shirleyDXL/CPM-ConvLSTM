import os
import json
import math
import time
import pickle
import math
import pandas as pd
import numpy as np
from collections import defaultdict



def get_point_list(shp):
    point_list_out = list()
    for d in shp:
        points =d["value"][0].strip().split(" ")
        points = [points[0], points[-1]]
        point_list_in = list()
        for point in points:
            str_point = [float(x) for x in point.split(",")]
            point_list_in.append(str_point)
        point_list_out.append(point_list_in)
    return point_list_out

"""
Get the start and end point of a path.

parameter:
point_list_out: the return of get_point_list, several GPS points list of a path.

return:
Tow GPS points
[[60.18275, 24.85485], [60.1827, 24.85476]]

"""
def get_end_point(point_list_out):
    point_list = list()
    end_point = list()
    for row in point_list_out:
        for col in row:
            point_list.append(col)
    for point in point_list:
        if point_list.count(point)==1:
            end_point.append(point)
    point = end_point[0]
    for row in point_list_out:
        if point in row:
            i = row.index(point)
    if i ==1:
        end_point.reverse()
    return end_point



def get_rwroad_firoad_shp(filename):
    savefile = "/home/dixl/project/traffic_project/data/road_keys.pkl"
    fi = open(savefile, 'rb')
    road_keys = pickle.load(fi)['road_keys']
    fi.close()
    shps = dict()
    with open(filename, "rb") as fi:
        data = json.load(fi)
        for rw in data['RWS'][0]['RW']:
            rwRoad = str(rw[u'DE']).replace(",","/").replace(" ","/")
            if rwRoad.isdigit() == False and rwRoad!="Kulosaarentie":
                for fi in rw['FIS'][0]['FI']:
                    fiRoad = fi['TMC']['DE'].replace(",","/").replace(" ","/")
                    qd = "1" if fi['TMC']['QD']=="+" else "0"
                    shp = fi["SHP"]
                    if (rwRoad,fiRoad,qd) in road_keys:
                        shps[(rwRoad,fiRoad,qd)] = shp
    return shps



def get_children(filename):

    shps = get_rwroad_firoad_shp(filename)

    key_no = dict()
    no_key = dict()
    i=0
    for key in shps.keys():
        key_no[key] = i
        no_key[i] = key
        i+=1

    data_dict = dict()
    for i in range(len(no_key)):
        data_dict[i] = get_end_point(get_point_list(shps[no_key[i]]))
    children_dict = defaultdict(list)

    for i in range(len(data_dict)):
        for j in range(len(data_dict)):
            if i==j:
                continue
            if data_dict[i][1] == data_dict[j][0] and data_dict[i][0] != data_dict[j][1] and no_key[i][2]==no_key[j][2]:
                children_dict[i].append(j)

    for i in range(len(data_dict)):
        if i not in children_dict.keys():
            children_dict[i]=[]
    return children_dict,key_no, no_key


def dict_to_json(data_dict, filename):
    jsObj = json.dumps(data_dict)
    fileObject = open(filename, 'w')
    fileObject.write(jsObj)
    fileObject.close()


def fillna(data):
    nan_position = np.where(np.isnan(data))
    for i in range(len(nan_position[0])):
        x=nan_position[0][i]
        y=nan_position[1][i]
        if x==(data.shape[0]-1):
            data.iloc[x,y] = data.iloc[x-1,y]
            continue
        if x==0:
            if np.isnan(data.iloc[x+1,y]):
                for j in range(x+2,data.shape[0]):
                    flag=0
                    if np.isnan(data.iloc[j,y])==False:
                        data.iloc[x,y] = (data.iloc[x-1,y] + data.iloc[j,y])/2
                        flag=1
                        continue
                    if flag==0:
                        data.iloc[x,y] = data.iloc[x-1,y]
                        continue
            else:
                data.iloc[x,y] = data.iloc[x+1,y]
                continue
        if np.isnan(data.iloc[x+1,y]):
            for j in range(x+2,data.shape[0]):
                flag=0
                if np.isnan(data.iloc[j,y])==False:
                    data.iloc[x,y] = (data.iloc[x-1,y] + data.iloc[j,y])/2
                    flag=1
                    break
                if flag==0:
                    data.iloc[x,y] = data.iloc[x-1,y]
        else:
            data.iloc[x,y] = (data.iloc[x-1,y] + data.iloc[x+1,y])/2
    return data

def get_dt_road(datafile, startTime, endTime, sort_name):
    date_road_dict = defaultdict(dict)
    with open(datafile, "r", encoding="utf-8") as fi:
        fi.readline()
        for row in fi:
            arr = row.split(",")
            if (arr[0]>=startTime) and (arr[0]<endTime):
                if arr[1] in congestion_dict and arr[2] in congestion_dict[arr[1]] and str(arr[5]) in congestion_dict[arr[1]][arr[2]]:
                    dt = arr[0][:-2]+"00"
                    if (arr[1],arr[2],arr[5]) in date_road_dict[dt].keys():
                        date_road_dict[dt][arr[1],arr[2],arr[5]].append(float(arr[7]))
                    else:
                        date_road_dict[dt][arr[1],arr[2],arr[5]] = [float(arr[7])]
    for dt in date_road_dict.keys():
        for road in date_road_dict[dt].keys():
            date_road_dict[dt][road]=np.mean(date_road_dict[dt][road])

    data_sort = pd.DataFrame.from_dict(date_road_dict, orient="index", columns=sort_name)


    date_index = pd.date_range(start=startTime,end=endTime,freq="min",closed='left')
    date_index =pd.DataFrame(date_index,columns=["rawTime"]).set_index("rawTime")

    data_sort = date_index.join(data_sort)

    data_sort = fillna(data_sort)

    return data_sort

def rad(d):
    return d * math.pi / 180.0

def cal_distance(point_1, point_2):
    lat_1 = point_1[0]
    lng_1 = point_1[1]
    lat_2 = point_2[0]
    lng_2 = point_2[1]

    radLat_1 = rad(lat_1)
    radLat_2 = rad(lat_2)
    a = rad(lat_1) - rad(lat_2)
    b = rad(lng_1) - rad(lng_2)

    s = 2 * math.asin(math.sqrt(math.pow(math.sin(a/2), 2) + math.cos(radLat_1)* math.cos(radLat_2)*math.pow(math.sin(b/2),2)))
    s = s*6378.137
    s = round(s*10000)/10
    return s
