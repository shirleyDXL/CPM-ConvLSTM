from util.grid import *
import json
import pickle
import calendar
import numpy as np
import pandas as pd
import util.PredictionUtil as pu

from util.ConstructUtil import *
from collections import defaultdict
from collections import OrderedDict
from treelib import Tree,Node
from util.params import Params as param




from model.convLSTM import *
#from CNN_LSTM import *
#from Model.C3D_convLSTM import *
from keras.models import load_model

def get_congestion_tree(treefile, children_dict):
    treefile = "/home/dixl/project/traffic_project/data/new_alltrees.json"
    with open(treefile, "r") as fi:
        trees = json.load(fi)
    congestion_pattern = trees['alltrees']
    new_congestion_pattern = list()
    for pattern in congestion_pattern:
        if pattern not in new_congestion_pattern:
            new_congestion_pattern.append(pattern)
    repeat_pattern = list()
    n = len(new_congestion_pattern)
    for i in range(n):
        for j in range(i+1, n):
            if set(new_congestion_pattern[j]).issubset(new_congestion_pattern[i]):
                if new_congestion_pattern[j] not in repeat_pattern:
                    repeat_pattern.append(new_congestion_pattern[j])

            if set(new_congestion_pattern[i]).issubset(new_congestion_pattern[j]):
                if new_congestion_pattern[i] not in repeat_pattern:
                    repeat_pattern.append(new_congestion_pattern[i])
    new_congestion_pattern_ = list()
    for pattern in new_congestion_pattern:
        if pattern not in repeat_pattern:
            new_congestion_pattern_.append(pattern)
    congestion_tree_list =  list()
    for nodes in new_congestion_pattern_:
        congestion_tree = Tree()
        used_nodes = defaultdict(list)
        used_nodes[0].append(nodes[0])
        congestion_tree.create_node(nodes[0],nodes[0])
        i=0
        while len(used_nodes[i]) != 0:
            for used_node in used_nodes[i]:
                for node in nodes:
                    if used_node in children_dict[node]:
                        if node not in list(congestion_tree.nodes.keys()):
                            congestion_tree.create_node(node,node, parent = used_node)
                            used_nodes[i+1].append(node)
            i+=1
        congestion_tree_list.append(congestion_tree)
    return congestion_tree_list



def get_min_max_GPS(data_dict):
    min_lon=1000
    min_lat=1000
    max_lon=0
    max_lat=0
    for key in data_dict:
        for row in data_dict[key]:
            for point in row:
                if point[0] > max_lat:
                    max_lat = point[0]
                if point[0] < min_lat:
                    min_lat = point[0]
                if point[1] > max_lon:
                    max_lon = point[1]
                if point[1] < min_lon:
                    min_lon = point[1]
    return min_lon, min_lat, max_lon, max_lat
def get_min_max_xy(data_dict):
    min_lon=1000
    min_lat=1000
    max_lon=0
    max_lat=0
    for point in data_dict:
#         point = data_dict[key]
        if point[0] > max_lat:
            max_lat = point[0]
        if point[0] < min_lat:
            min_lat = point[0]
        if point[1] > max_lon:
            max_lon = point[1]
        if point[1] < min_lon:
            min_lon = point[1]
    return min_lon, min_lat, max_lon, max_lat


def make_road_grid(n_c_p, shps, no_key):
    data_dict = dict()
    for no in n_c_p:
        data_dict[no_key[no]] = shps[no_key[no]]
    min_lon, min_lat, max_lon, max_lat = get_min_max_GPS(data_dict)

    len_x = cal_distance([min_lat, min_lon], [max_lat, min_lon])
    len_y = cal_distance([min_lat, min_lon], [min_lat, max_lon])
    k = int((len_x+len_y)/200)
    if k==0:
        k=2
    ag = AreaGrid(k, min_lon, min_lat, max_lon, max_lat)

    keys = list(data_dict.keys())
    raw_road_grid = dict()
    for key in keys:
        grid_no = defaultdict(list)
        for row in data_dict[key]:
            for point in row:
                no = ag.gid(point)
                grid_no[no].append(point)
        raw_road_grid[key] = grid_no

    road_grid = defaultdict(list)
    for key in raw_road_grid:
        for no in raw_road_grid[key]:
            flag = 0
            if len(raw_road_grid[key][no]) > 2:
                flag=1
                road_grid[key].append(no)
        if flag==0:
            for no in raw_road_grid[key]:
                if no not in road_grid[key]:
                    road_grid[key].append(no)
    keys = list(data_dict.keys())
    grid_road = defaultdict(list)
    for key in keys:
        for row in data_dict[key]:
            for point in row:
                no = ag.gid(point)
                str_key = key
                if str_key not in grid_road[no]:
                    grid_road[no].append(str_key)

    return road_grid,grid_road,ag,k


def judge_grid(road1_name, road2_name,shps):
    shp1 = get_end_point(shps[road1_name])
    shp2 = get_end_point(shps[road2_name])
    if shp1[1] ==shp2[0]:
        lat_d = abs(shp1[0][0]-shp2[1][0])
        long_d = abs(shp1[0][1]-shp2[1][1])
        if lat_d > long_d:#2,7
            if shp1[0][0]>shp2[1][0]:
                return 8
            else:
                return 2
        else:
            if shp1[0][1]>shp2[1][1]:
                return 4
            else:
                return 6
    else:
        lat_d = abs(shp1[1][0]-shp2[0][0])
        long_d = abs(shp1[1][1]-shp2[0][1])
        if lat_d > long_d:#2,7
            if shp1[1][0]>shp2[0][0]:
                return 8
            else:
                return 2
        else:
            if shp1[1][1]>shp2[0][1]:
                return 4
            else:
                return 6


def judge_position(road1, road2, road1_name, road2_name, k,shps):

    if len(road1)>2:
        r1 = road1[1]
    else:
        r1 = road1[0]
    if len(road2)>2:
        r2 = road2[-2]
    else:
        r2 = road2[0]

    if int(r1/k)!=int(r2/k):#不同行
        ##
        if int(r1/k) > int(r2/k):#进入7,8,9
            if (r1-r2)%k<int(k/2):
                return 7
            elif (r1-r2)%k>int(k/2):
                return 9
            else:
                return 8
        else:
            if (r1-r2)%k<int(k/2):
                return 3
            elif (r1-r2)%k>int(k/2):
                return 1
            else:
                return 2
    else:#同行
        if r1>r2:
            return 4
        elif r1<r2:
            return 6
        else:
            #同grid比较
            return judge_grid(road1_name, road2_name,shps)

def get_position(road1, road2, road1_name, road2_name,road_relative_position,k,road_grid,shps):
    position_id = judge_position(road_grid[road1_name],road_grid[road2_name], road1_name,road2_name,k,shps)
    raw_position = road_relative_position[road1_name]
    if position_id in [1,2,3]:
        raw_position =  (raw_position[0], raw_position[1]+1)
    if position_id in [3,6,9]:
        raw_position =  (raw_position[0]+1, raw_position[1])
    if position_id in [1,4,7]:
        raw_position =  (raw_position[0]-1, raw_position[1])
    if position_id in [7,8,9]:
        raw_position =  (raw_position[0], raw_position[1]-1)
    return raw_position


def get_position_2(road1_name,position_id, road_relative_position):
    raw_position = road_relative_position[road1_name]
    if position_id in [1,2,3]:
        raw_position =  (raw_position[0], raw_position[1]+1)
    if position_id in [3,6,9]:
        raw_position =  (raw_position[0]+1, raw_position[1])
    if position_id in [1,4,7]:
        raw_position =  (raw_position[0]-1, raw_position[1])
    if position_id in [7,8,9]:
        raw_position =  (raw_position[0], raw_position[1]-1)
    return raw_position

def get_level_node(tree, n):
    level_nodes = list()
    nodes = tree.all_nodes()
    for node in nodes:
        if tree.level(node.identifier) == n:
            level_nodes.append(node.identifier)
    return level_nodes


def get_reshape_data(testy, predicted, rows, cols):

    testy_ = testy.reshape(testy.shape[0],rows*cols)
    predicted_=predicted.reshape(predicted.shape[0],rows*cols)

    return testy_, predicted_


def get_data(congestion_tree, k, no_key, road_grid,date_road_dict,shps,workday = True):
    start_road = no_key[congestion_tree.root]
    road_relative_position = {start_road:(0,0)}
    position_road = {(0,0):start_road}
    height = congestion_tree.depth()
    for i in range(1,height+1):
        level_nodes = get_level_node(congestion_tree, i)
        for node in level_nodes:
            road1_name = no_key[congestion_tree.parent(node).tag]
            road2_name = no_key[node]
            raw_position = get_position(road_grid[road1_name],road_grid[road2_name], road1_name,road2_name,road_relative_position,k,road_grid,shps)
            n=0
            flag = 0
            while n<10:
                if raw_position not in position_road:
                    road_relative_position[road2_name] = raw_position
                    position_road[raw_position] = road2_name
                    flag = 1
                    break
                else:
                    roadx_name = position_road[raw_position]
                    raw_position = get_position(road_grid[roadx_name],road_grid[road2_name], roadx_name,road2_name, road_relative_position,k,road_grid,shps)
                n+=1
            if n>=10:
                for id_ in [1,2,3,4,6,7,8,9]:
                    raw_position = get_position_2(road1_name,id_,road_relative_position)
                    if raw_position not in position_road:
                        road_relative_position[road2_name] = raw_position
                        position_road[raw_position] = road2_name
                        flag = 1
                        break
    min_y, min_x, max_y, max_x = get_min_max_xy(position_road)
    rows = max_y - min_y + 1
    cols = max_x - min_x + 1
    number_road = dict()
    road_number = dict()
    for position in position_road:
        no = (position[1]-min_y)*cols + (position[0]-min_x) + 1
        number_road[no] =position_road[position]
        road_number[position_road[position]] = no
    mat_congestion = defaultdict(dict)
    nos =[i for i in range(1,rows*cols+1)]
    for dt in date_road_dict:
        weekday = calendar.weekday(int(dt[:4]),int(dt[5:7]),int(dt[8:10]))
        if workday:
            if weekday>4:
                continue
        else:
            if weekday<5:
                continue
        for no in nos:
            if no in number_road and number_road[no] in date_road_dict[dt]:
                mat_congestion[dt][no] = date_road_dict[dt][number_road[no]]
            else:
                mat_congestion[dt][no] = 0
    nos_reverse = [i for i in range(rows*cols,0,-1)]
    data_sort = pd.DataFrame.from_dict(mat_congestion, orient="index", columns=nos_reverse)
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
#     date_index = pd.date_range(start=startTime,end=endTime,freq="min",closed='left')
    date_index = list(mat_congestion.keys())
    date_index =pd.DataFrame(date_index,columns=["rawTime"]).set_index("rawTime")
    data_sort = date_index.join(data_sort)
    data_sort = fillna(data_sort)
    return data_sort,rows,cols,road_number,number_road


def reshape_data(data, split_num=1440, rows=39, cols=23, length=20):

    raw_data = data.values.reshape(data.shape[0],rows,cols,1)
    train_data = raw_data[:-split_num]
    test_data = raw_data[-split_num:]
    train_X, train_y = pu.split_X_y(train_data, length)
    test_X, test_y = pu.split_X_y(test_data, length)

    return train_X, train_y, test_X, test_y


def train(data_sort, rows, cols, number_road):

    train_X, train_y, test_X, test_y = reshape_data(data_sort, param.split_num, rows, cols, param.sequence_length)
    conv_model, conv_testy, conv_predicted = convLSTM_model(train_X, train_y, test_X, test_y, param.unit, param.epochs)
#    train_y = train_y.reshape(train_y.shape[0],train_y.shape[1]*train_y.shape[2])
#    test_y=test_y.reshape(test_y.shape[0],test_y.shape[1]*test_y.shape[2])
#    cnn_model, cnn_testy, cnn_predicted = CNN_LSTM_model(train_X, train_y, test_X, test_y,1)
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error

    testy_, predicted_ = get_reshape_data(conv_testy, conv_predicted, rows, cols)
#    testy_, predicted_ = cnn_testy, cnn_predicted
    true_y = list()
    predict_y = list()
    for n in range(testy_.shape[0]):
        for no in list(number_road.keys()):
            true_y.append(testy_[n][-no])
            predict_y.append(predicted_[n][-no])
    mse = mean_squared_error(true_y, predict_y)
    mae = mean_absolute_error(true_y, predict_y)
    print("mse=",mse)
    print("mae=",mae)
    return mse,mae