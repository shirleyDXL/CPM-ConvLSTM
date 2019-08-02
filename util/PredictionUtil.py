import pandas as pd
import numpy as np
from collections import defaultdict

def split_X_y(rawData, sequence_length=20):

    results = list()
    for i in range(len(rawData) - sequence_length+1):
        results.append(rawData[i:i+sequence_length])
        
    results = np.array(results)
    train_X = results[:,:-1]
    train_y = results[:,-1]
    
    return train_X, train_y

def get_road_grid():
    filename = "/home/dix1/Traffic_prediction/Data/1535707658.json"
    shps = get_rwroad_firoad_shp(filename)
    for key in shps.keys():
        shps[key] = get_point_list(shps[key])
        
    keys = list(shps.keys())
    road_grid = defaultdict(list)
    for key in keys:
        for row in shps[key]:
            for point in row:
                no = ag.gid(point)
                if key not in road_grid:
                    road_grid[key].append(no)
                else:
                    if no not in road_grid[key]:
                        road_grid[key].append(no)
                        
    return road_grid

def get_reshape_data(testy, predicted, k=30):
    road_grid = get_road_grid()
    
    rno_list = list()
    for key in road_grid:
        for rno in road_grid[key]:
            if rno not in rno_list:
                rno_list.append(rno)
                
    testy_ = testy.reshape(testy.shape[0],k*k)
    predicted_=predicted.reshape(predicted.shape[0],k*k)
    
    return testy_, predicted_
    
def grid_mse_mae_part(testy, predicted, k=30):
    testy_, predicted_ = get_reshape_data(testy, predicted, k=30)
    
    ty=list()
    py = list()
    for i in range(len(testy_)):
        for rno in rno_list:
            ty.append(testy_[i][-rno])
            py.append(predicted_[i][-rno])  
            
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error
    mse = mean_squared_error(ty, py)
    mae = mean_absolute_error(ty, py)
    print("mse=",mse)
    print("mae=",mae)
    
def grid_mse_mae_road(testy, predicted, k=30):   
    testy_, predicted_ = get_reshape_data(testy, predicted, k=30)
    test_date_index = date_index = pd.date_range(start="2018-10-05 00:19:00",end=endTime,freq="min",closed='left')
    true_y = list()
    predict_y = list()
    for i in range(len(test_date_index)):
        dt = test_date_index[i].strftime("%Y-%m-%d %H:%M:%S")
        for road_key in date_road_dict[dt]:
            sum_congestion = list()
            for road_no in road_grid[road_key]:
                sum_congestion.append(predicted_[i][-road_no])
            true_y.append(date_road_dict[dt][road_key])
            predict_y.append(np.mean(sum_congestion))
            
    mse = mean_squared_error(true_y, predict_y)
    mae = mean_absolute_error(true_y, predict_y)
    print("mse=",mse)
    print("mae=",mae)