# Import all the required libraries
import requests
from pathlib import Path
import io
import pandas as pd
import numpy as np
import shapely
from shapely.geometry import Polygon
import shapely.geometry
import geopandas as gpd
from shapely import wkt
import matplotlib.colors as mcolors
import utm
import os,json

def process_json_files(path_to_json):
    """
    This function reads all the JSON files in a given directory and performs some data processing.

    Parameters:
    path_to_json (str): The path to the directory where the JSON files are located.

    Returns:
    pandas.DataFrame: A DataFrame containing the processed data.
    """
    data_list = []
    for file_name in [file for file in os.listdir(path_to_json) if file.endswith('.json')]:
        with open(os.path.join(path_to_json, file_name)) as json_file:
            data = json.load(json_file)

    # Flatten data & include lat & lng
    df_netsed_list = pd.json_normalize(data, record_path=['property_data'], meta=['lat', 'lng'])

    # Drop columns with NA values for camera parameters which later would be used for filtering POI images
    df_netsed_list = df_netsed_list.dropna(subset=['computed_geometry.coordinates', 'camera_parameters', 'computed_compass_angle']).reset_index(drop=True)

    # Split columns into multiple columns
    split_camera_parameters = pd.DataFrame(df_netsed_list['camera_parameters'].tolist(), columns=['camera_parameters (Focal Lenght)', 'camera_parameters (K1)', 'camera_parameters (K2)'])
    # Join split columns back to original DataFrame
    df_netsed_list = pd.concat([df_netsed_list, split_camera_parameters], axis=1)
    # Split columns into multiple columns
    split_coordinates = pd.DataFrame(df_netsed_list['computed_geometry.coordinates'].tolist(), columns=['Computed coordinatess Lng', 'Computed coordinatess Lat'])
    # Join split columns back to original DataFrame
    df_netsed_list = pd.concat([df_netsed_list, split_coordinates], axis=1)

    df_netsed_list = df_netsed_list.dropna(subset=['Computed coordinatess Lng', 'Computed coordinatess Lat', 'camera_parameters (Focal Lenght)']).reset_index(drop=True)

    df_netsed_list['captured_at_date'] = pd.to_datetime(df_netsed_list['captured_at'], unit='ms').dt.floor('S') # remove second
    df_netsed_list['captured_at_year'] = df_netsed_list['captured_at_date'].dt.year
    df_netsed_list['captured_at_month'] = df_netsed_list['captured_at_date'].dt.month
    df_netsed_list['captured_at_day'] = df_netsed_list['captured_at_date'].dt.day
    df_netsed_list['captured_at_hour'] = df_netsed_list['captured_at_date'].dt.time
    mapillary_df = df_netsed_list.drop(columns=['computed_geometry.type', 'camera_parameters', 'computed_geometry.coordinates', 'captured_at', 'geometry.type'])

    data_list.append(mapillary_df)

    # Concatenate the list of dataframes into one
    final_df = pd.concat(data_list, axis=0)

    return final_df

def spatial_join_mapillary_safegraph(csv_path, df_netsed_list, d=55):
    safegraph_df = pd.read_csv(csv_path)

    gdf1 = gpd.GeoDataFrame(
        safegraph_df,
        geometry=gpd.points_from_xy(safegraph_df['longitude'], safegraph_df['latitude']),
        crs='epsg:4326'
    )

    gdf2 = gpd.GeoDataFrame(
        df_netsed_list,
        geometry=gpd.points_from_xy(df_netsed_list['lng'], df_netsed_list['lat']),
        crs='epsg:4326'
    )

    gdf_final = gdf2.sjoin(gdf1, how="left").reset_index(drop=True)

    gdf_final['building_poly'] = gdf_final['polygon_wkt'].apply(wkt.loads)
    gdf_final.drop('polygon_wkt', axis=1, inplace=True)

    gdf_final['focal_px'] = gdf_final['camera_parameters (Focal Lenght)'] * (gdf_final[['width', 'height']].max(axis=1))
    gdf_final['teta'] = 2 * np.arctan(gdf_final['width'] / (2 * gdf_final['focal_px']))

    gdf_final['c_utm_lat'] = utm.from_latlon(gdf_final['Computed coordinatess Lat'], gdf_final['Computed coordinatess Lng'], 18, 'T')[0]
    gdf_final['c_utm_lng'] = utm.from_latlon(gdf_final['Computed coordinatess Lat'], gdf_final['Computed coordinatess Lng'], 18, 'T')[1]

    gdf_final['ax_utm_lng'] = gdf_final['c_utm_lng'] + (d * np.cos(np.radians(gdf_final['computed_compass_angle']) - (gdf_final['teta'] / 2)))
    gdf_final['ay_utm_lat'] = gdf_final['c_utm_lat'] + (d * np.sin(np.radians(gdf_final['computed_compass_angle']) - (gdf_final['teta'] / 2)))

    gdf_final['bx_utm_lng'] = gdf_final['c_utm_lng'] + (d * np.cos(np.radians(gdf_final['computed_compass_angle']) + (gdf_final['teta'] / 2)))
    gdf_final['by_utm_lat'] = gdf_final['c_utm_lat'] + (d * np.sin(np.radians(gdf_final['computed_compass_angle']) + (gdf_final['teta'] / 2)))

    gdf_final['ax_lng'] = utm.to_latlon(gdf_final['ay_utm_lat'], gdf_final['ax_utm_lng'], 18, 'T')[1]
    gdf_final['ay_lat'] = utm.to_latlon(gdf_final['ay_utm_lat'], gdf_final['ax_utm_lng'], 18, 'T')[0]

    gdf_final['bx_lng'] = utm.to_latlon(gdf_final['by_utm_lat'], gdf_final['bx_utm_lng'], 18, 'T')[1]
    gdf_final['by_lat'] = utm.to_latlon(gdf_final['by_utm_lng'], gdf_final['bx_utm_lng'], 18, 'T')[0]
    # Trangular based on camera position and two vertices
    for i, item in gdf_final.iterrows():
        gdf_final.loc[i, 'triangular_polygon'] = Polygon(
            [(item['Computed coordinatess Lng'], item['Computed coordinatess Lat']),
             (item['ax_lng'], item['ay_lat']),
             (item['bx_lng'], item['by_lat'])
             ])

    gdf_final['triangular_polygon'] = gdf_final['triangular_polygon'].apply(str)
    triangular_polygon = gdf_final['triangular_polygon'].map(shapely.wkt.loads)
    gdf_final = gdf_final.drop('triangular_polygon', axis=1)
    gdf_final = gpd.GeoDataFrame(gdf_final, crs="EPSG:4326", geometry=triangular_polygon)

    return gdf_final


def intersect(x):
    return x[0].intersects(x[1])


def area(x):
    return round((x[0].intersection(x[1]).area / x[0].area) * (100), 2)


def area_point(x):
    return x[0].intersection(x[1])
def process_gdf(gdf):
# Consider the two columns have polygon to check their intersection
    gdf3 = gdf.filter(items=['geometry', 'building_poly'])
    gdf['intersection_area'] = gdf3.apply(area, axis=1)
    gdf['filter'] = gdf3.apply(intersect, axis=1)
    gdf['Coordinates'] = list(zip(gdf['lat'], gdf['lng']))
    gdf['image_ids'] = gdf['id'].astype('str') + '.jpg'

    return gdf