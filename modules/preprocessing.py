##############################################################################
################ Author: Poshan Niraula ######################################
################ Date: 2021-01-18 ############################################


######################dependencies############################################
# 1. numpy
# 2. pandas
# 3. matplotlib
# 4. os
# 5. datetime
# 6. geopandas
###############################################################################



import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
from datetime import datetime as dt
import geopandas as gpd

#######################################################################
###functoin that takes covid data, number of lag days, and 
###the boundary file to create the number of cases in neighboring regions, 
###shifted cases in lag days, cumulative cases, 
#######################################################################

def preprocessing(data: str, geometry: str, lag_days: int) -> pd.DataFrame:
    '''
         Assumptions in data
             fieldname for daily cases - 'daily_new_infected'
             fieldname for daily deaths - 'daily_new_death'
             fieldname for unique healthzone code - 'health_zone_code'
             fieldname for date - 'date'
             encoding - latin_1
         
         assumptions in geometry
             fieldname for unique healthzone code - 'hzcode'
    
         Parameters
         ----------
         data : STRING
             FILE PATH OF THE CSV FILE CONTAINING DAILY INFECTED DATA.
         geometry : STRING
             FILE PATH OF THE .SHP FILE CONTAINING GEOMETRY OF THE REGION OF STUDY.
         lag_days : INT
             NUMBER OF DAYS TO COMPUTE THE INFECTION CASES T- DAYS.
    
         Returns
         -------
         DATAFRAME ADDING FROM THE INPUT DF - SHIFTED CASES IN LAG DAYS, 
         CUMULATIVE CASES AND AVERAGE OF NEIGHBORS 
    
     '''
    
    #######################################################################
    ########read the csv file
    #######################################################################
    print('...............reading daily infection data...................')
    df = pd.read_csv(data, encoding = 'latin_1')
    
    
    #######################################################################
    ###shift each healthzones data by lag_days days
    #######################################################################
    print('...................shifting the cases' + 'by'+ str(lag_days) + ' days')
    df['shifted_cases'] = df.groupby('health_zone_code')['daily_new_infected'].shift(lag_days).fillna(0)
    
    
    #######################################################################
    ###cumulative cases
    #######################################################################
    print('...................cummulative sum till t-1 ................')
    df['cum_cases'] = df.groupby(['health_zone_code'])['daily_new_infected'].apply(lambda x: x.cumsum())
    df['cum_cases'] = df['cum_cases'] - df['daily_new_infected']
    
    
    #######################################################################
    ####average number of cases in the neighbors
    #######################################################################
    print('...........computing the average neighbor cases.............')
    
    boundary_df = gpd.read_file(geometry)
    
    boundary_df["NEIGHBORS"] = None  # add NEIGHBORS column
    
    for index, municipality in boundary_df.iterrows():   
        # get 'not disjoint' countries
        neighbors = boundary_df[~boundary_df.geometry.disjoint(municipality.geometry)].hzcode.tolist()
        # remove own name from the list
        neighbors = [ hz for hz in neighbors if municipality.hzcode != hz ]
        # add names of neighbors as NEIGHBORS value
        boundary_df.at[index, "NEIGHBORS"] = ", ".join(neighbors)
    
    
    
    ##in infected data select only regions in the boundary files
    geometry_regions = boundary_df['hzcode'].unique()
    df = df[df['health_zone_code'].isin(geometry_regions)]
    
    
    ##set the index as the hzcode
    boundary_df['hzcode'] = [int(h) for h in boundary_df['hzcode']]
    boundary_df.set_index('hzcode', inplace = True)
       
    
    ###remove every column except the NEIGHBORS column and index of course :P
    remove_cols = []
    for c in boundary_df.columns:
        if (c != 'NEIGHBORS'):
            remove_cols.append(c)
    
    boundary_df.drop(remove_cols, axis = 'columns', inplace = True)
    
    
    ###join the df with neighbors
    df.set_index('health_zone_code', inplace = True)
    df = df.join(boundary_df)
    
    
    
    #################enhanchment needed ##################################
    ################slow in computation###################################
    
    ##compute the cases in the neighbors in the given date and healthzoens
    #compute average cases in teh neighbors
    
    #adding a column to add the cases of neighbors on that day
    df['cases_neighbors']= None
    df['deaths_neighbors'] = None
    
    df['cases_neighbors'] = df['cases_neighbors'].astype('object')
    df['deaths_neighbors'] = df['deaths_neighbors'].astype('object')
    
    #empty list for cases in the neighbors and mobility in neighbors
    cases_neighbors = []
    deaths_neighbors = []
    
    for index, row in df.iterrows():
        #neighbor list for each row
        print(row['NEIGHBORS'])
        neighbors = [int(n) if (n!= '') else index for n in row['NEIGHBORS'].split(', ')]
        date = row['date']
    
        #append a list with 0 for regions with no neighbors
        if (len(neighbors) == 0):
            cases_neighbors.append([0])
            deaths_neighbors.append([0])
    
        else:
            #subsetting the df by neighbors and date
            sub_df = df.loc[neighbors,:][['date','daily_new_infected', 'daily_new_death']]
            cases_neighbors.append(sub_df[sub_df['date'] == date]['daily_new_infected'].to_list())
            deaths_neighbors.append(sub_df[sub_df['date'] == date]['daily_new_death'].to_list())
        # df_cases_neighbors.at[index, 'cases_neighbors'] = cases_neighbors
        # df_cases_neighbors.at[index, 'deaths_neighbors'] = cases_neighbors
    df['cases_neighbors'] = cases_neighbors
    df['deaths_neighbors'] = deaths_neighbors
    
    print('.................preprocessing done...............................')
    
    return df.copy()
    



