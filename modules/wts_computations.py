##############################################################################
################ Author: Poshan Niraula ######################################
################ Date: 2021-01-18 ############################################
########### Data sources######################################################
########### Barcelona Supercomputing center - Mobility matrices ##############
########### Daily covid cases - Castilla Y Leon dataportal ###################
##############################################################################


######################dependencies############################################
# 1. numpy
# 2. pandas
# 3. matplotlib
# 4. os
# 5. datetime
###############################################################################



######################################################import libraries required
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
from datetime import datetime, timedelta
# import geopandas as gpd


##"Processed_data/CYL_cases_neighbors_shifted.csv" - cyl_covid_path
##"covid_data/Cyl_hzone_covid_01March_20Nov.csv" - daily_cases_path

def wts_computations(mobility_dir: str, cyl_covid_path: str, lag_days: int) -> pd.DataFrame:
    print("weights computations")

########################################################defining the dataframes
    cyl_covid = pd.read_csv(cyl_covid_path, encoding = 'latin_1')
    
    ##############################################################################
    ###########################mobility weights computations######################
    ##############################################################################
    
    
    ####setting dates in the df
    cyl_covid["date1"] = pd.to_datetime(cyl_covid["date"])
    
    #remove the time part split by space
    cyl_covid["date1"] = [str(d).split(" ")[0] for d in cyl_covid["date1"]]
    
    #all the unique dates in the dataframe
    dates_cases = cyl_covid["date1"].unique()
    
    #############################computing the infection ratio with the population
    
    #sort the values by healthzones and date
    
    ####fix this---unnamed 0 from the preprocessing in the healthzone
    cyl_covid['health_zone_code'] = cyl_covid['Unnamed: 0']
    
    cyl_covid.sort_values(['health_zone_code','date'], inplace=True)
    
    ##groupby healthzone and rolling sum for the detla + 1 day
    df_new = cyl_covid.groupby(["health_zone_code"])[['daily_new_infected']].rolling(min_periods = 1, 
                                                                                          window = lag_days + 1).sum()
    
    ##reset index to remove the healthzones from the index
    df_new.reset_index(inplace=True)
    
    
    #subtract the same day to account other days then today
    cyl_covid['last_ndays_cases'] = [c-d for c, d in zip(df_new['daily_new_infected'], 
                                                              cyl_covid['daily_new_infected'])]
    
    ##last 4 days cummulative
    cyl_covid["Inf_w"] = [inf/pop for inf, pop in zip(cyl_covid["last_ndays_cases"],
                                               cyl_covid["total_pop"])]
    
    
    ###################################################Weight total from infections
    cyl_covid["total_w"] = cyl_covid["Inf_w"]
    
    ##################################################Mobility weights computations
    #for the unique list of 245 healthzones
    one_day_df = pd.read_csv(mobility_dir + "/20200301.csv", index_col="hzcode_orig")
    one_day_df.head()
    indices = one_day_df.index.unique()
    
    
    #################################function that adds mobility for the delta days
    def sum_mobility(date: str) -> np.ndarray:
      d = datetime.strptime(date, "%Y-%m-%d")
      sum_mobility = np.zeros((245,245))
      for i in range(1, lag_days + 1):
        d_i = d - timedelta(days = i)
        x = str(d_i)[:10].split('-')
        fname  = x[0] + x[1] + x[2] + '.csv' 
        # print(fname)
        file_mob = os.path.join(mobility_dir, fname)
        if os.path.exists(file_mob):
          df_tmp = pd.read_csv(file_mob, index_col="hzcode_orig")
          df_tmp.sort_index(axis="columns", inplace=True)
          df_tmp.sort_index(axis="rows", inplace=True)
          sum_mobility = np.add(sum_mobility, df_tmp.values)
      return (sum_mobility)
    
    ############################################# select 245 healthzones out of 247
    cyl_covid.set_index('health_zone_code', inplace = True)
    daily_cases_df_new = cyl_covid[cyl_covid.index.isin(indices)]
    daily_cases_df_new.sort_index()
    
    ###############################################computing weights for each date
    indices = indices
    # indices_names = [map_municode_name[c] for c in indices]
    wt_list = []
    df_weight_new = pd.DataFrame(columns=("date","hzcodes","weights"))
    
    for d in dates_cases:
      if d in dates_cases:
        ##summing up the mobility for last lag_days
        df_tmp = sum_mobility(d)
        mob_matrix = df_tmp
      else:
        print("no mobility for date", d)
      
      ###infections ratio for the data   
      daily_cases_df_tmp = daily_cases_df_new[daily_cases_df_new.date1 == d].sort_index()
      weight_matrix = daily_cases_df_tmp["total_w"].values.astype("float32")
      
      # multiply mob_matrix with weight_matrix to get the result for each date
      res_weight = np.matmul(mob_matrix, weight_matrix)
    
      # appending to list
      wt_df1 = pd.DataFrame()
      wt_df1["weights"] = res_weight
      wt_df1["date"] = [d] * len(indices)
      wt_df1["hzcodes"] = indices
      df_weight_new = df_weight_new.append(wt_df1)
      wt_list.append(res_weight)
    
    
    ##################################join the computed weights with other datasets
    # setting the same index (date-hzcode) for the join
    cyl_covid["date_hzcode"] = [d + "-" + str(hz) for d, hz in zip(cyl_covid["date1"], cyl_covid.index)]
    cyl_covid.set_index("date_hzcode", inplace=True)
    
    # setting the same index (date-hzcode) for the join
    df_weight_new["date_hzcode"] = [d1 + "-" + str(hz1) for d1, hz1 in zip(df_weight_new["date"], 
                                                                           df_weight_new["hzcodes"])]
    df_weight_new.set_index("date_hzcode", inplace=True)
    df_weight_new.drop(["date","hzcodes"], axis="columns", inplace=True)

    
        
    print('........joining.............')
    # joing
    cyl_covid_wt = cyl_covid.join(df_weight_new, lsuffix="_cases")
#     print(cyl_covid_wt.columns)
    cyl_covid_wt = cyl_covid_wt[cyl_covid_wt["date1"].isin(dates_cases)]
#     print(cyl_covid_wt.head())
    
    ##remove na
    cyl_covid_wt = cyl_covid_wt.dropna(subset=["weights"])
    
    print(cyl_covid_wt.columns)
    ###remove unnecessary columns
#     cyl_covid_wt.drop(["Datecases","dateshift","hzone_code.1","NEIGHBORS","cases_neighbors",
#                        "deaths_neighbors", "d_zbs"], axis = "columns", inplace=True)
    
    ####compute the day of the week
    cyl_covid_wt['date'] = [i.split('-')[0]+"-"+i.split('-')[1]+"-"+i.split('-')[2] 
                            for i in cyl_covid_wt.index]
    cyl_covid_wt['date'] = pd.to_datetime(cyl_covid_wt['date'])
    cyl_covid_wt["dow"] = cyl_covid_wt['date'].dt.dayofweek
    cyl_covid_wt.drop("date", axis="columns", inplace=True)
    
    return cyl_covid_wt
    
    print('finished')
    #############################################################################
    #################################end########################################
    #############################################################################
