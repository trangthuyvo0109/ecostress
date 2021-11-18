# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 13:06:02 2020

@author: tvo
"""
'''
This scipt is create to pre-process ECOSTRESS. The tasks include:
        
    - Extracting all decode and masked out images to real figures, at which can justify the images's 
    quality by eyes
    - Extract stats info of the good quality images 
    - Decoding Cloud Mask product to remove cloudy and water pixels from the region
    - Snipcode for Set Null of the decoded Cloud Mask and original images in Arcpy 
    - Running Zonal Stats using gdal
    - Let the algorythm running for each images: from linear_np_test_ver02_linux.py
    
    + Note: Besides, if we wish to run parallel processing, we can uncomment the main function at the end
    of the code. 
    
'''



def show_image(folder):
    '''
    # Export all ECOSTRESS images to images file, to easily detect the potential image
    
    Example:
        folder = '/nas/rhome/tvo/py_code/aes509/data/ECOSTRESS/'

        show_image(folder)
    '''


    import rasterio
    import cmocean  
    import os
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm
    from matplotlib.ticker import MaxNLocator
    from matplotlib.colors import Normalize
    from matplotlib import cm
    import matplotlib.pyplot as plt
    import numpy as np
    import numpy.ma as ma
    import matplotlib
    
    matplotlib.rcParams['font.size'] = 20
    matplotlib.rcParams['legend.fontsize'] = 20
    matplotlib.rcParams['axes.labelweight'] = 'bold'
    matplotlib.rcParams['axes.titleweight'] = 'bold'
    matplotlib.rcParams['axes.labelsize'] = 20
    
    
    for file in os.listdir(folder):
        try:
            if 'SDS_LST_doy' in file:
                src = rasterio.open(folder+file)
                array = src.read(1)     
                # Define levels of ticks and tick intervals
                array_masked = ma.masked_where(array == 65535, array) # 65535 is null value
                # array_masked = ma.masked_where(array == 0, array)
                tick_min=np.nanmin(array_masked*0.02)
                tick_max=np.nanmax(array_masked*0.02)
#                if tick_max - tick_min > 50:
#                    tick_min = tick_max - 50
                        
                
                #tick_min = 285
                #tick_max = 320
                tick_interval = (tick_max - tick_min)/100
                levels = MaxNLocator(nbins=(tick_max - tick_min)/tick_interval).tick_values(tick_min, tick_max)
                cmap = plt.get_cmap(cmocean.cm.thermal)
                #cmap = plt.get_cmap('coolwarm')
                norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
                
                
                fig, ax = plt.subplots(figsize=(15,15))
                map_color = plt.imshow(array_masked*0.02,cmap=cmap,norm=norm)
                fig.colorbar(map_color)
                fig.savefig(folder+'/images/'+file)
                
                src.close()
                
            else:
                pass
            
        except:
            print(file)
            
            
            
def info_good_images(folder_good_images,title):
    '''
    Function to extract the stats of good iamges 
    
    Example:
        folder_good_images = '//uahdata/rhome/py_code/aes509/data/ECO_large_04092020/images/good_images'
        title = 'ECOSTRESS observations with good quality (non-clear sky)'
        eco_stats_good= info_good_images(folder_good_images,title)
        
    '''
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib
    import datetime
    import sys
    matplotlib.rcParams['font.size'] = 20
    matplotlib.rcParams['legend.fontsize'] = 20
    matplotlib.rcParams['axes.labelweight'] = 'bold'
    matplotlib.rcParams['axes.titleweight'] = 'bold'
    matplotlib.rcParams['axes.labelsize'] = 20
    
    
    if sys.platform == 'linux':
        eco_stats = pd.read_csv('/nas/rhome/tvo/py_code/aes509/data/ECO_large_04092020/ECO2LSTE-001-Statistics.csv')
        
    else:
        eco_stats = pd.read_csv('//uahdata/rhome/py_code/aes509/data/ECO_large_04092020/ECO2LSTE-001-Statistics.csv')
    
    
    good_images = []
    for file in os.listdir(folder_good_images):
        if file.endswith('.tif'):   
            good_images.append(file.split('.tif')[0].split('SDS_')[-1].split('_aid')[0])
        
    # Search for code froom stats of ecostress
    eco_stats_good = eco_stats.loc[eco_stats['Doy'].isin(good_images)]
 
    # Export to .csv
    eco_stats_good.to_csv(folder_good_images+'/'+'stats_good_images.csv')
    
    # Insert year column
    eco_stats_good['Year'] = pd.to_numeric(eco_stats_good['Date'].str[0:4])
    eco_stats_good['Day'] = pd.to_datetime(eco_stats_good['Date'].str[0:19],format='%Y-%m-%d %H:%M:%S')
    
    eco_stats_good = eco_stats_good.set_index('Day')
    
    # Convert UTC to local time zone
        # Concert from UTC to local time
    from dateutil import tz
    
    # Get zone
    from_zone = tz.gettz('UTC')
    to_zone = tz.gettz('America/New_York')
    
    # Tell the datetime object that it is in UTC
    # date time objects are 'naive by default'
    eco_stats_good.index = eco_stats_good.index.tz_localize(from_zone)
    
    # Convert to local time zone
    eco_stats_good.index = eco_stats_good.index.tz_convert(to_zone) 
    
    # Filter day time and night time
    day_time_1 = datetime.time(18,0,0)
    day_time_2 = datetime.time(6,0,0)
    
    eco_stats_good_day = eco_stats_good.loc[(eco_stats_good.index.time > day_time_2)&
                                            (eco_stats_good.index.time < day_time_1)]
    
    eco_stats_good_night = eco_stats_good.loc[(eco_stats_good.index.time > day_time_1)|
                                            (eco_stats_good.index.time < day_time_2)]
    
    # Plotting
    fig, ax = plt.subplots()
    ax.plot(eco_stats_good_day.index,eco_stats_good_day['Maximum'],'o',label='Day time Observation',color='red')
    ax.plot(eco_stats_good_night.index,eco_stats_good_night['Maximum'],'o',
             label='Night time Observation',color='black')
    
    plt.xlabel('Date Time (in UTC)')
    plt.ylabel('Maximum LST (K)')
    
    plt.legend()
    plt.title(title)

    
    
 

    return eco_stats_good
           


def cloud_mask(file_cloudmask,
               file_LST,
               outFile):
    '''
    Function to decode cloud mask image and assign cloudy pixels nan values
    
    
    Example:
    file_cloudmask = 'C:/Users/tvo/Documents/urban_project/ECO_large_04092020_cloudmask/ECO2CLD.001_SDS_CloudMask_doy2019264063150_aid0001.tif'
    file_LST = '//uahdata/rhome/Blogging/ECOSTRESS_decode/ECO2LSTE.001_SDS_LST_doy2019278182133_aid0001.tif'
    outFile = file.split('/ECO2')[0] + '/decoced/'+file.split('cloudmask/')[-1].split('.tif')[0]+'_decoded.tif'
    cloud_mask(file, outFile)
    


    '''


    import os
    import gdal
    import rasterio
    import cmocean  
    import os
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm
    from matplotlib.ticker import MaxNLocator
    from matplotlib.ticker import FixedLocator
    from matplotlib.colors import Normalize
    from matplotlib import cm
    import matplotlib.pyplot as plt
    import numpy as np
    import numpy.ma as ma
    import matplotlib
    
    print('Decoding '+file_LST)
    # read cloud mask
    dataset = gdal.Open(file_cloudmask)
    
    data = dataset.ReadAsArray()
    
    # read EOSTRESS LST or Emis data
    dataset_eco = gdal.Open(file_LST)
    
    data_eco = dataset_eco.ReadAsArray()  
    
    # Get row,col
    [cols, rows] = data.shape

    
    # Maske out nan values
    data = np.ma.masked_where(data == -999, data)
    data_eco = np.ma.masked_where(data == -999, data_eco)
    
    # decode the cloud mask 8-bit
    qa_decode = np.vectorize(np.binary_repr)(data,width=8)
    
    
    # Only take first 3-bit to decode
    cloud = np.frompyfunc(lambda x:x[-3:],1,1)(qa_decode)
    
    # Seach all 111 and replace with 1
    cloud[cloud == '111'] = 1
    
    # Search the rest and replace with 0
    cloud[(cloud != 1)&(cloud != np.nan)] = 0
    
    # Convert to int value
    cloud = cloud.astype('int')

    # Testing with original data and decoded cloud mask
    fig, ax = plt.subplots()
    plt.imshow(data_eco,
               cmap='jet',
               vmin = data_eco[data_eco != 0].min(),
               vmax = data_eco[data_eco != 0].max(),
               alpha=0.7)
    plt.imshow(cloud, cmap='gray',alpha=0.5)
    
    # Exporting to a new decoded image
    # Driver
    driver = gdal.GetDriverByName('GTiff')
    
    # Output
    outdata = driver.Create(outFile, rows, cols, 1,  gdal.GDT_UInt16)
    
    # Set geotransform same as input
    outdata.SetGeoTransform(dataset.GetGeoTransform())
    
    # Set projection same as input
    outdata.SetProjection(dataset.GetProjection())
    
    # Write image
    outdata.GetRasterBand(1).WriteArray(cloud)
    
    # Set NoData
    outdata.GetRasterBand(1).SetNoDataValue(-999)
    
    
    # Save to disk
    outdata.FlushCache()
    
    outdata = None
    dataset = None
    
    
    return cloud   



'''
# Snipt code for SetNull in Arcpy
import os 
 ####### Version 2 : 05.10.2020 #########
import os
import arcpy
from arcpy.sa import *

folder_good_images = 'C:/Users/tvo/Documents/urban_project/ECO_large_04092020/images/good_images/clear_sky'

folder_ori = 'C:/Users/tvo/Documents/urban_project/ECO_large_04092020/'

# Due to the problem of ECOSTRESS decode products,
# Pick up one cloud mask image what looks perfectly 
# separating land/water for all images



in_condi_raster = 'C:/Users/tvo/Documents/urban_project/ECO_large_04092020_cloudmask/decoced/ECO2CLD.001_SDS_CloudMask_doy2019059153557_aid0001_decoded.tif'
code_list = []
for item in os.listdir(folder_good_images):
    if item.endswith('.tif'):
        code = item.split('SDS_LST_')[-1].split('_aid')[0]
        code_list.append(code)

for item in code_list:   
    for file_ori in os.listdir(folder_ori):
        if 'LST_'+item in file_ori:
            in_false_raster_LST = folder_ori  + file_ori

        if 'EmisWB_'+item in file_ori:
            in_false_raster_emiswb = folder_ori + file_ori
        
    try:
        print(in_false_raster_LST,in_false_raster_emiswb)
        # Excute set Null function 
        outNull_LST = SetNull(in_condi_raster,in_false_raster_LST,'VALUE = 1')
        outNull_emiswb = SetNull(in_condi_raster,in_false_raster_emiswb,'VALUE = 1')
        
        # Excuute Raster Calculator for scalling
        outNull_LST_scaled = outNull_LST * 0.02
        outNull_emiswb_scaled = outNull_emiswb * 0.002 + 0.49
        
        
        # Save the output 
        outNull_LST_scaled.save(folder_ori+'set_null/'+in_false_raster_LST.split('ECO_large_04092020/')[-1].split('.tif')[0]+'_scaled_cloudmask.tif')
        outNull_emiswb_scaled.save(folder_ori+'set_null/'+in_false_raster_emiswb.split('ECO_large_04092020/')[-1].split('.tif')[0]+'_scaled_cloudmask.tif')
    
    except Exception as e:
        print(e)
        pass
       
# End snipcode 
        
'''


'''
Snip code 

# Note : this snipd code using only ONE cloud mask for all images 
    #####
    
import arcpy
from arcpy.sa import *
import os
    
def set_null(in_condi_raster,in_false_raster,out_folder):


    print('Setting Null....')
    #outfolder = 'C:/Users/tvo/Documents/urban_project/ECO_large_04092020/scaled/'    
    # Excute set Null function 
    outNull = SetNull(in_condi_raster,in_false_raster,'VALUE = 1')
    #outNull_emiswb = SetNull(in_condi_raster,in_false_raster_emiswb,'VALUE = 1')
 
    # Save the output 
    outNull.save(out_folder+in_false_raster.split('/')[-1].split('.tif')[0]+'_scaled_cloudmasked.tif')
    #outNull_emiswb.save(out_folder+in_false_raster_emiswb.split('/')[-1].split('.tif')[0]+'_scaled_cloudmasked.tif')
            
        



# End snipcode 
folder_cloud = 'C:/Users/tvo/Documents/urban_project/ECO_large_04092020_cloudmask/decoced/'
folder_ori = 'C:/Users/tvo/Documents/urban_project/ECO_large_04092020/'
out_folder = folder_ori+'set_null/'


for file in os.listdir(folder_cloud):
    if file.endswith('.tif'):
        in_condi_raster = folder_cloud+file # Conditional raster: cloud masked images 

for file_ori in os.listdir(folder_ori):
    print(file_ori)
    if file_ori.endswith('.tif'):
        code = file_ori.split('_doy')[1].split('_aid')[0] # Extract doy code of the current image 
        
        in_false_raster = folder_ori  + file_ori
    
        print(in_false_raster)    
        set_null(in_condi_raster,in_false_raster,out_folder)
    else:
        pass
            

   
'''

def collect_result(result):
    global results
    
    results.append(result)             
    
def zonal(input_zone_polygon,input_value_raster,path,path_fig): 
#    input_zone_polygon = '/nas/rhome/tvo/py_code/aes509/data/whole_nyc_70m/AOI_grid_shp/AOI_grid.shp'
#    
#    path = '/nas/rhome/tvo/py_code/aes509/data/ECO_large_zonal_test/'
#    path_fig  = path+'out_figure/'
#    shp_path = '/nas/rhome/tvo/py_code/aes509/data/whole_nyc_70m/AOI_grid_shp/AOI_grid.shp'
#    df_geo = geopandas.read_file(shp_path)
#    raster_folder = '/nas/rhome/tvo/py_code/aes509/data/ECO_large_04092020/set_null'
    

    # Using rasterstats
    import rasterstats
    from rasterstats import zonal_stats
    import geopandas
    import os.path
    import os
    
    print('Zonal Stast for '+input_value_raster.split('input_value_raster/')[-1])
    try:
        print(path+input_value_raster.split('set_null/')[-1].split('.tif')[0]+'.csv')
        if os.path.isfile(path+input_value_raster.split('set_null/')[-1].split('.tif')[0]+'.csv') is True:
            pass
        else:
        
            stats = zonal_stats(input_zone_polygon, input_value_raster,
                        #stats=['count','mean','min', 'max', 'median', 'majority', 'sum'],
                        all_touched=True,
                        )
            
            import pandas as pd
            import matplotlib.pyplot as plt
            df = pd.DataFrame.from_dict(stats)
            df.to_csv(path+input_value_raster.split('set_null/')[-1].split('.tif')[0]+'.csv')
            
            df_geo = geopandas.read_file(input_zone_polygon)
            df_concat = pd.concat([df_geo,df],axis=1).dropna()
            
            print('Mapping the results .... ')
            fig, ax = plt.subplots()
            df_concat.plot(column='mean',ax=ax)
            fig = plt.gcf()
            fig.savefig(path_fig+input_value_raster.split('set_null/')[-1].split('.tif')[0]+'.png')
            
    except Exception as e:
        print(e)
        
        
        
    
def map_geopandas(shp_file, zonal_file, path_fig):
    '''
    
    Example:

        shp_file = '//uahdata/rhome/py_code/aes509/data/ECO_large_zonal_test/AOI_grid/AOI_grid'
        zonal_file = '//uahdata/rhome/py_code/aes509/data/ECO_large_zonal_test/emiswb_2018236120944.dbf'
        path_fig = '//uahdata/rhome/py_code/aes509/data/ECO_large_zonal_test/out_figure/'
        # Linux:
    
        map_geopandas(shp_file,zonal_file,path_fig)
    '''
    
    #Mapping with geopandas 

    print('Mapping results of Zonal Stats')
    import geopandas 
    import pandas as pd
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import matplotlib
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    import matplotlib.gridspec as gridspec
    import numpy as np
    import cmocean
    import simpledbf
    from simpledbf import Dbf5

    
    
    matplotlib.rcParams['savefig.pad_inches'] = 0
    matplotlib.rcParams['font.size'] = 20
    matplotlib.rcParams['legend.fontsize'] = 20
    matplotlib.rcParams['axes.labelweight'] = 'bold'
    matplotlib.rcParams['axes.titleweight'] = 'bold'
    matplotlib.rcParams['axes.labelsize'] = 20
    
    
    # Read data
    #df_geo = geopandas.read_file(shp_file)
    df_geo = pd.read_pickle(shp_file)
    df_zonal = Dbf5(zonal_file).to_dataframe()
    
    
    df_concat = pd.concat([df_geo,df_zonal],axis=1).dropna()
    
    cmap = plt.get_cmap(cmocean.cm.thermal)
    # # create the colorbar
    vmin = df_concat.MEAN.min()
    vmax = df_concat.MEAN.max()
    norm1 = colors.Normalize(vmin=vmin, vmax=vmax)
    cbar1 = plt.cm.ScalarMappable(norm=norm1, cmap=cmap)
    

    # Mapping
    proj = ccrs.AlbersEqualArea()
    # Set bounds:
    bounds_lat = [19.233772,19.073554]
    bounds_lon = [17.253122,17.358485]
    ll_proj = ccrs.Geodetic()
    fig, ax1 = plt.subplots(figsize=(17,12),
    subplot_kw=dict(projection=proj))
    
    

    
#    _ = ax1.set_extent(bounds_lon+bounds_lat,ll_proj)
#    ax1.outline_patch.set_linewidth(0)
#    _ = ax2.set_extent(bounds_lon+bounds_lat,ll_proj)
#    ax2.outline_patch.set_linewidth(0)
    
    plot = df_concat.plot(column='MEAN',ax=ax1,cmap=cmap)
    
    # Position of colorbar
    axins1 = inset_axes(ax1,
                        width="50%",  # width = 50% of parent_bbox width
                        height="3%",  # height : 5%
                        loc='lower center',
                        #bbox_to_anchor = (0,0,1,1),
                        bbox_transform=ax1.transAxes,
                        borderpad=0 # Distance to/from border of frame
                        )
    
    

    
    
    # add colorbar

    ax1_colorbar = fig.colorbar(cbar1,ax=ax1,
    orientation='horizontal',cax=axins1)
    
    
    fig.savefig(path_fig+zonal_file.split('.dbf')[0]+'_map.png')

    
if __name__ == "__main__":

    '''
    Function to apply paralleling 
    
    '''
    import ecostress_preprocessing
    from ecostress_preprocessing import *
    from multiprocessing import Pool
    import multiprocessing as mp
    import os
    import os.path
    import sys
    
    if sys.platform == 'linux':
    
        # Extract name of best quality images:
        folder_good_images = '/nas/rhome/tvo/py_code/aes509/data/ECO_large_04092020/images/good_images/clear_sky/'
        title = 'ECOSTRESS observations with best quality (clear sky)'
        eco_stats_good= info_good_images(folder_good_images,title)
        eco_good_images_name = eco_stats_good.Doy.str[4:].tolist()
        
        
        
        # Paralleling new version version
        pool = mp.Pool(mp.cpu_count())
        
    
        # First paralleing for decoding cloud masked images 
        folder = '/nas/rhome/tvo/py_code/aes509/data/ECO_large_04092020_cloudmask/'
        #folder = '//uahdata/rhome/py_code/aes509/data/ECO_large_04092020_cloudmask/'
        
        results = []
    
        for item in eco_good_images_name:
            for file in os.listdir(folder):
                if item in file:
                    outFile = folder + 'decoced/'+file.split('cloudmask/')[-1].split('.tif')[0]+'_decoded.tif'
                    
                    if os.path.isfile(outFile) is False: 
                        #cloud_mask(folder + file,outFile)
                        print(outFile)
                        
                        pool.apply_async(cloud_mask, 
                                     args=(folder + file,
                                           outFile), 
                                     callback=collect_result)
                                     
                        
            
            
        # Close Pool and let all processes complete
        pool.close()
        pool.join()# postpones the execution of next line of code until all processes in the queue are done.
     
        
        print(results)
    

    
    
#if __name__ == "__main__":
##def main():
#    '''
#    Function to apply paralleling 
#    
#    '''
#    import gdal_read_write
#    from gdal_read_write import *
#    from multiprocessing import Pool
#    import multiprocessing as mp
#    import os
#    import os.path
#    import sys
#    
#    if sys.platform == 'linux':
#        # Paralleling new version version
#        pool = mp.Pool(mp.cpu_count())
#        
#    
#        # First paralleing for decoding cloud masked images 
#        #raster_folder = '/nas/rhome/tvo/py_code/aes509/data/ECO_large_04092020/set_null'
#        raster_folder = '/nas/rhome/tvo/py_code/aes509/data/ECO_large_04092020/georeferenced/set_null/'
#        input_zone_polygon = '/nas/rhome/tvo/py_code/aes509/data/whole_nyc_70m/AOI_grid_shp/AOI_grid.shp'  
#        #path = '/nas/rhome/tvo/py_code/aes509/data/ECO_large_zonal_test/'
#        path = '/nas/rhome/tvo/py_code/aes509/data/ECO_large_zonal_test/georeferenced/'
#        path_fig  = path+'out_figure/'
#        #folder = '//uahdata/rhome/py_code/aes509/data/ECO_large_04092020_cloudmask/'
#        
#        results = []
#        print('Trang')
#        
#        for file in os.listdir(raster_folder):
#            if file.endswith('.tif'):
#                input_value_raster = raster_folder + file
#                print(file)
#            
#                        
#                pool.apply_async(zonal, 
#                                     args=(input_zone_polygon,
#                                           input_value_raster,
#                                           path,
#                                           path_fig), 
#                                     callback=collect_result)
#                                     
#                        
#    
#        # Close Pool and let all processes complete
#        pool.close()
#        pool.join()# postpones the execution of next line of code until all processes in the queue are done.
#              
#    
#
#

















