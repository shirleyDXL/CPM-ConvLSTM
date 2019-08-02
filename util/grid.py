# import pandas as pd
# import cPickle as pickle
import numpy as np
import pandas as pd

lonStep_1m = 0.0000105
latStep_1m = 0.0000090201


class RoadGrid:
    def __init__(self, label, grid_size):
        length = grid_size * latStep_1m
        width = grid_size * lonStep_1m
        self.length = length
        self.width = width

        def orginal_plot(label):
            # top right
            tr = np.max(label, axis=0)
            tr[0] += 25 * lonStep_1m
            tr[1] += 25 * latStep_1m
            # bottom left
            bl = np.min(label, axis=0)
            bl[0] -= 25 * lonStep_1m
            bl[1] -= 25 * latStep_1m

            return bl[0], tr[0], bl[1], tr[1]

        xl, xr, yb, yt = orginal_plot(label)
        self.xl = xl
        self.xr = xr
        self.yb = yb
        self.yt = yt
        gridSet = set()
        grid_dict = {}
        for pos in label:
            lon = pos[0]
            lat = pos[1]

            m = int((lon - xl) / width)
            n = int((lat - yb) / length)
            if not grid_dict.has_key((m, n)):
                grid_dict[(m, n)] = []
            grid_dict[(m, n)].append((lon, lat))
            gridSet.add((m, n))
        gridlist = list(gridSet)

        grid_center = [tuple(np.mean(np.array(lonlat_list), axis=0)) for (i, j), lonlat_list in grid_dict.items()]

        self.gridlist = gridlist

        self.grids = [(xl + i[0] * width, yb + i[1] * length) for i in grid_dict.keys()]
        self.grid_center = grid_center
        self.n_grid = len(self.grid_center)

    def transform(self, label, sparse=True):
        def one_hot(idx, n):
            a = [0] * n
            a[idx] = 1
            return a

        grid_pos = [self.gridlist.index((int((i[0] - self.xl) / self.width), int((i[1] - self.yb) / self.length))) for i
                    in label]
        if sparse:
            grid_pos = np.array([one_hot(x, len(self.gridlist)) for x in grid_pos], dtype=np.int32)
        return grid_pos

    def average_value(self, features, labels, column_names=None):
        grid_pos = self.transform(labels.values, False)
        assert features.shape[0] == labels.shape[0] == len(grid_pos)
        ave_val = [0] * (max(grid_pos) + 1)
        for i in range(0, max(grid_pos) + 1):
            t = []
            for idx, pos in enumerate(grid_pos):
                if pos == i:
                    t.append(features.loc[idx, column_names])
            if len(t) > 0:
                ave_val[i] = pd.concat(t, axis=1).T.mean(axis=0)
        for idx, pos in enumerate(grid_pos):
            features.loc[idx, column_names] = ave_val[pos]

        return features


class AreaGrid:
    def __init__(self, k, min_lon, min_lat, max_lon, max_lat):
        self.min_lon = min_lon
        self.min_lat = min_lat
        self.max_lon = max_lon
        self.max_lat = max_lat
        self.k = k
        self.grids = []
        # lonStep_1m = 0.0000105
        # latStep_1m = 0.0000090201
        self.lon_offset = (self.max_lon - self.min_lon) / self.k
        self.lat_offset = (self.max_lat - self.min_lat) / self.k
        for i in range(self.k):
            for j in range(self.k):
                id = i * self.k + j + 1
                lon = self.min_lon + j * self.lon_offset
                lat = self.min_lat + i * self.lat_offset
                grid = (id, ((lat, lon), (lat + self.lat_offset, lon + self.lon_offset)))
                self.grids.append(grid)

    def gid(self, pt):
        lat, lon = pt
        if self.min_lon <= lon <= self.max_lon and \
                self.min_lat <= lat <= self.max_lat:
            i = int((lat - self.min_lat) / self.lat_offset)
            if i == self.k:
                i=i-1
            j = int((lon - self.min_lon) / self.lon_offset)
            if j == self.k:
                j=j-1
            return i * self.k + j + 1
        else:
            return -1

    def within(self, pt, grid):
        lat, lon = pt
        (ld_lat, ld_lon), (ru_lat, ru_lon) = grid
        return True if (ld_lon <= lon <= ru_lon) and (ld_lat <= lat <= ru_lat) else False

    def dump_boundary(self, filename):
        f_out = open(filename, 'w')
        for id, ((x1, y1), (x2, y2)) in self.grids:
            f_out.write('%.6f,%.6f,%.6f,%.6f\n' % (x1, y1, x2, y2))
        f_out.close()
