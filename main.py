import argparse
from model.supervisor import *
from util.params import Params as param

def main(args):
    startTime = args.startTime
    endTime = args.endTime

    filename = args.assistfile
    raw_shps = get_rwroad_firoad_shp(filename)
    shps = dict()
    for key in raw_shps.keys():
        shps[key] = get_point_list(raw_shps[key])
    pkl_file = open(args.datafile, 'rb')
    date_road_dict = pickle.load(pkl_file)

    children_dict, key_no, no_key = get_children(filename)

    with open("test_result.txt", "w") as fi:
        m = 0
        congestion_tree_list = get_congestion_tree(args.treefile, children_dict)

        for congestion_tree in congestion_tree_list:
            print(m)
            n_c_p = list(congestion_tree.nodes.keys())
            road_grid,grid_road,ag,k = make_road_grid(n_c_p, shps, no_key)
            data_sort,rows,cols,road_number,number_road = get_data(congestion_tree,k, no_key,road_grid,date_road_dict,shps,param.workday)
            mse,mae = train(data_sort, rows, cols,number_road)
            fi.write(str(mse)+","+str(mae)+"\n")
        m=m+1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafile', default='/home/dixl/project/traffic_project/data/all_date_road_dict.pkl', type=str,
                        help='datafile')
    parser.add_argument('--assistfile', default="/home/dixl/project/traffic_project/data/1535707658.json", type=str,
                        help='get child dict')
    parser.add_argument('--treefile', default="/home/dixl/project/traffic_project/data/new_alltrees.json", type=str,
                        help='stctree')
    parser.add_argument('--startTime', default='2018-09-01 00:00:00', type=str, help='start time')
    parser.add_argument('--endTime', default='2018-10-06 00:00:00', type=str, help='start time')
    args = parser.parse_args()
    main(args)



