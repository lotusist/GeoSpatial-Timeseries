# # Import modules

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob
import os


# # Data concat
print("Collecting data...")
# setting the path for joining multiple files
ais = os.path.join("./220630 AIS", "*.csv")
lte = os.path.join("./220630 LTE-M", "*.csv")
# list of merged files returned
ais = glob.glob(ais)
lte = glob.glob(lte)

# joining files with concat and read_csv
ais_concat = pd.concat(map(pd.read_csv, ais), ignore_index=True)
lte_concat = pd.concat(map(pd.read_csv, lte), ignore_index=True)

#data type processing
ais_concat = ais_concat.astype({'szMsgSendDT':'int'})
lte_concat = lte_concat.astype({'szMsgSendDT':'int'})
ais_concat['szMsgSendDT'] = ais_concat['szMsgSendDT']/1000
lte_concat['szMsgSendDT'] = lte_concat['szMsgSendDT']/1000
ais_concat['szMsgSendDT'] = pd.to_datetime(ais_concat['szMsgSendDT'], format = "%Y%m%d%H%M%S")
lte_concat['szMsgSendDT'] = pd.to_datetime(lte_concat['szMsgSendDT'], format = "%Y%m%d%H%M%S")
ais_concat.index = ais_concat['szMsgSendDT']
lte_concat.index = lte_concat['szMsgSendDT']
print("Done!")
print("AIS : ", len(ais_concat['szSrcID'].unique()), "LTE : ", len(lte_concat['szSrcID'].unique()),"\n")

# # Data Filtering

print("Filtering shipments...")
'''shipment filtering'''

#group by each ship
ais_group = ais_concat.groupby('szSrcID')
lte_group = lte_concat.groupby('szSrcID')


#filter ships out where dSOG is nearly 0
ais_move = ais_group.filter(lambda group : group['dSOG'].max()>2)
lte_move = lte_group.filter(lambda group : group['dSOG'].max()>2)


#find common ids
ais_ids = set(ais_move['szSrcID'].values)
lte_ids = set(lte_move['szSrcID'].values)
com_ids = list(ais_ids&lte_ids)

#filter common ids only
ais_com = ais_move[ais_move['szSrcID'].isin(com_ids)].sort_index()
lte_com = lte_move[lte_move['szSrcID'].isin(com_ids)].sort_index()

print("Done!")
print("AIS : ", len(ais_move['szSrcID'].unique()), "LTE : ", len(lte_move['szSrcID'].unique()),"\n")


print("Filtering paths...")

def add_col(dataframe : pd.DataFrame, latlon_cut: float):
  '''
  INPUT 
    - dataframe: target dataframe to cleanse
    - latlon_cut: a float num 0~1.(higher than .5 recommended) an argument to get quantile of lat delta or long delta

  OUTPUT
    - dataframe with new columns
      - flag: an int score indicating the point may be an outlier in lat long 
  '''
  data = dataframe.copy()

  data['lat_delta'] = abs(data['dLat'] - data['dLat'].shift(1, fill_value=data['dLat'][0]))
  data['lon_delta'] = abs(data['dLon'] - data['dLon'].shift(1, fill_value=data['dLon'][0]))

  lat_cut = data['lat_delta'].quantile(q=latlon_cut, interpolation='nearest')
  lon_cut = data['lon_delta'].quantile(q=latlon_cut, interpolation='nearest')

  data['flag'] = data.apply(lambda x: int(x.lat_delta > lat_cut) + int(x.lon_delta >lon_cut), axis='columns')

#   print("Num of 'flag'\n", data['flag'].value_counts())

  return data

def extract_route(dataframe: pd.DataFrame):
  '''
  used FLAG .

  INPUT
    - dataframe: target dataframe (or, the output of function 'add_col')

  OUTPUT
    - list of dataframes, each of which is one cleansed route points
  '''

  data = dataframe.copy()
  route_list = list()
  valid_time = list(data.loc[data['flag'] == 2]['szMsgSendDT'])
  for i in range(len(valid_time)-1):
    one_route = data.loc[(data['szMsgSendDT'] >= valid_time[i])
                          & (valid_time[i+1] >= data['szMsgSendDT'])]
    if not one_route.empty and (one_route['lon_delta']!=0).all() :  
      route_list.append(one_route)
  if not route_list: return None #없는 경우 return None
  concat_route = pd.concat(route_list, axis = 0)
  concat_route = concat_route.loc[concat_route['flag']!=2]  
  return concat_route

def filter(df):
  df = add_col(df, 0.9)
  new_df = extract_route(df)
  return new_df


'''points filtering'''

ais_filtered = ais_com.groupby('szSrcID').apply(lambda group: filter(group))

lte_filtered = lte_com.groupby('szSrcID').apply(lambda group: filter(group))

ais_filtered.index = ais_filtered.droplevel('szSrcID')
lte_filtered.index = lte_filtered.droplevel('szSrcID')
ais_filtered.index = ais_filtered['szMsgSendDT']
lte_filtered.index = lte_filtered['szMsgSendDT']

print("Done!")
print("AIS : ", len(ais_filtered['szSrcID'].unique()), "LTE : ", len(lte_filtered['szSrcID'].unique()),"\n")


print("Prediction...")
# # Trajectory Prediction

def find_dist(input, output):
    x1 = input['dLon']
    x2 = output['dLon']
    y1 = input['dLat']
    y2 = output['dLat']
    x1, x2, y1, y2 = map(np.radians, [x1, x2, y1, y2])
    dx = x2- x1
    dy = y2-y1
    a = np.sin(dy/2.0)**2 + np.cos(y1) * np.cos(y2) * np.sin(dx/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    dist = 6367 * c
    return dist


def find_next(data):
    x_pred = [data['dLon'][0]]
    y_pred = [data['dLat'][0]]
    output = pd.DataFrame(data[['szMsgSendDT', 'szSrcID', 'dLon', 'dLat', 'dCOG', 'dSOG']])

    for t in range(len(data)-1): 
        dt = ((data['szMsgSendDT'][t+1] - data['szMsgSendDT'][t]).seconds)/3600     #unit conversion(s -> hr)
        x_t1 = data['dLon'][t] + (dt/60)* np.sin(np.radians(data['dCOG'][t])) * data['dSOG'][t] / np.cos(np.radians(data['dLat'][t]))
        y_t1 = data['dLat'][t] + (dt/60)* np.cos(np.radians(data['dCOG'][t])) * data['dSOG'][t]
        x_pred.append(x_t1)
        y_pred.append(y_t1)

    output['dLon'] = x_pred
    output['dLat'] = y_pred

    return output        


def find_err(input, output):
    #L1
    output['x_err'] = abs(input['dLon'] - output['dLon'])
    output['y_err'] = abs(input['dLat'] - output['dLat'])
    #L2 sqrt
    output['dist2D'] = np.sqrt(np.square(output['x_err'])+np.square(output['y_err']))
    #Harvestine 3D 
    output['dist3D'] = find_dist(input, output)
    return output


def prediction(df):
    pred = find_next(df)
    output = find_err(df, pred)
    return output


"""prediction for all shipments"""

ais_output = ais_filtered.groupby('szSrcID',group_keys=True).apply(lambda group : prediction(group))
lte_output = lte_filtered.groupby('szSrcID',group_keys=True).apply(lambda group : prediction(group))

print("Done!\n")

print("Output to CSV")
ais_output.to_csv("ais_prediction.csv")
lte_output.to_csv("lte_prediction.csv")
