import pandas as pd
import geopandas as gpd
import os

###class mobility conversion
class MobilityConversion:
    def __init__(self, mob, intersection, id_req, id_mob, output):
        '''
        .gz files inside the mob folder assumed
        
        Parameters
        ----------
        mob : STRING
            FOLDER PATH CONTAINING THE DAILY MOBILITY MATRICES.
        intersection:STRING
            FILE PATH CONTAINING THE INTERSECTION OF MOBILITY BOUNDARY AND REQUIRED BOUNDARY
        id_req : STRING
            UNIQUE ID IN THE SHAPEFILE FOR THE REQUIRED SHAPES.
        id_mob: STRING
            UNIQUE ID IN THE SHAPEFILE FOR THE MOBILITY SHAPES.
        output : STRING
            FOLDER PATH TO SAVE THE NEW MOBILITY MATRICES.

        Returns
        -------
        None.

        '''

        # self.shp_mob = gpd.read_file(shp_mob).to_crs(epsg=4326)
        # self.shp_req = gpd.read_file(shp_req).to_crs(epsg=4326)
        self.id_req = id_req
        self.id_mob = id_mob
        self.output = output
    
        self.mobility_files = []
        for subdir, dirs, files in os.walk(mob):
            for file in files:
              if file.endswith(".gz"):
                self.mobility_files.append(os.path.join(subdir, file))
        
        
        self.intersection = gpd.read_file(intersection).to_crs({'init': 'epsg:3857'})
        self.mob_zones = self.intersection[id_mob].unique()
        self.req_zones = self.intersection[id_req].unique()
        '''
        1. compute the area with same ID of the available boundary
        2. compute the percentage of area occupied by each polygon 
        
        '''
        
        
        self.intersection["area_float"] = self.intersection['geometry'].area
        self.intersection["total_area"] = None

        
        for index, row in self.intersection.iterrows():
            area = self.intersection[self.intersection[self.id_mob] == row[self.id_mob]]["area_float"].sum()
            self.intersection.at[index, "total_area"] = area
        
        self.intersection["percentage_area"] = [a/ta for a, ta in zip(self.intersection["area_float"], self.intersection["total_area"])]
        
        self.intersection.set_index(self.id_mob, inplace = True)
        
        print(f"there are {len(self.mob_zones)} polygons")
        print(f"Converting to {len(self.req_zones)} polygons")
        
        
    def get_mobilityfiles(self):
        '''
        

        Returns
        -------
        TYPE: LIST
            FULL FILENAME OF MOBILITY MATRICES.

        '''
        return self.mobility_files
        
    def get_shapemob(self):
        '''
        

        Returns
        -------
        TYPE
            RETURN THE GEODATAFRAME OF THE BOUNDARY WITH THE GIVEN MOBILITY DATA.

        '''
        return self.shp_mob
    
    
    def get_shapereq(self):
        '''
        

        Returns
        -------
        TYPE
            RETURN THE GEODATAFRAME OF THE BOUNDARY WITH THE REQUIRED SHAPE.

        '''
        return self.shp_req
    
    
    def get_intersection(self):
        '''
        

        Returns
        -------
        TYPE
            RETURN THE GEODATAFRAME OF THE INTERSECTION OF TWO DATAFRAMES.

        '''
        return(self.intersection)
    
    def get_output(self):
        '''
        

        Returns
        -------
        TYPE
            OUTPUT PATH OF THE CONVERTED MOBILITIES.

        '''
        return(self.output)
        
    def convert(self):
        '''
        Convert the mobility data in the tabular format to the matrices
        Transform the data in one areal units to another

        Returns
        -------
        None.

        '''
        for file in self.mobility_files:
            print(file)
            
            #read the csv
            df = pd.read_csv(file, sep="|", compression = "gzip")
            
            #selecting only for CYL municipalities
            df = df[(df["origen"].isin(self.mob_zones))&(df["destino"].isin(self.mob_zones))]
            df_tmp = df.copy()
            
            #grouping by the origen destino summing all the viajes
            df_tmp_grouped = df_tmp.groupby(["origen","destino"])[["viajes"]].sum()
        
            # joining with the intersection for destination
            df_tmp_grouped.reset_index(inplace=True)
            df_tmp_grouped.set_index("destino", inplace=True)
            df_tmp_grouped_join_intersection = df_tmp_grouped.join(self.intersection)
            df_tmp_grouped_join_intersection.head()
        
            #saving the destination for future purpose
            df_tmp_grouped_join_intersection["destino"] = df_tmp_grouped_join_intersection.index
        
            # joining wiht the intersection again for origin
            df_tmp_grouped_join_intersection.set_index("origen", inplace = True)
            df_tmp_join_again = df_tmp_grouped_join_intersection.join(self.intersection, lsuffix = "_orig", rsuffix = "_dest")
            df_tmp_join_again.head()
        
        
            df_tmp_join_again["weighted_viajes"] = df_tmp_join_again["viajes"] * df_tmp_join_again["percentage_area_orig"] * df_tmp_join_again["percentage_area_dest"]
            
            df_semifinal = df_tmp_join_again.groupby([self.id_req + "_orig",self.id_req + "_dest"])[["weighted_viajes"]].sum()
        
            #pivot
            df_tmp_matrix = pd.pivot_table(df_semifinal, values = "weighted_viajes", index=self.id_req + "_orig", columns=self.id_req + "_dest", aggfunc = sum, fill_value=0)
        
            #reindex to add all the municipalities in CYL
            df_tmp_matrix = df_tmp_matrix.reindex(index=self.req_zones, columns=self.req_zones, fill_value=0)
        
            #dont forget to sort please
            df_tmp_matrix.sort_index(axis="columns", inplace=True)
            df_tmp_matrix.sort_index(axis="rows", inplace=True)
        
            #printing the shape of the matrix
            print(df_tmp_matrix.shape)
        
            #save
            filename = os.path.join(self.output, file.split("/")[-1][:8]+".csv")
            df_tmp_matrix.to_csv(filename)
            print('.................conversion completed......................')
            