from shiny import App, ui, render, reactive
import os
import json
import pandas as pd
import rasterio
from rasterio.mask import mask
import numpy as np
from shapely.geometry import shape
import shapely.ops
from pyproj import Transformer
from datetime import datetime, timedelta
import plotly.express as px
from shinywidgets import output_widget, render_widget
import geopandas as gpd
from folium.plugins import Draw
import leafmap.foliumap as leafmap
import pathlib
import plotly.graph_objs as go
from faicons import icon_svg
import subprocess
import subprocess
import os
import gzip
import shutil
from datetime import datetime
import glob
from osgeo import gdal, osr, ogr
from joblib import Parallel, delayed
from rasterio import features  # Make sure this line is present
from rasterio.features import geometry_mask
from shapely.geometry import box
from tqdm import tqdm
import asyncio
from io import StringIO


shapefile_path = f"downscaling/shp/SNIC_30000_V2.shp"
gdf = gpd.read_file(shapefile_path)


# Lecture des fichiers de géodonnées
bassin = gpd.read_file("shp/MitidjaOuest.shp")
bassingeo = gpd.read_file("geojson/MitidjaOuest.geojson")

# Chemins fixes vers les répertoires de fichiers
aet_dir = 'outputs/AET_dekad_brut_m'
chirps_dir = 'outputs/Chirps_dekad_brut_m'
diff_dir = 'outputs/difirence_dekad_brut_m'
mask_tif_path = 'outputs/mask_irrig_season1_2019.tif'
path_full_basin = "geojson/3.geojson" 
# Fonction pour déterminer la saison en fonction du Month
def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Fall"
########## Calcul Missing Data ################
def analyze_directory_dates(directory):
    """Analyser les dates de début/fin et les dekads manquantes dans un répertoire."""
    files = glob.glob(os.path.join(directory, '*.tif'))
    if not files:
        return None, None, []
    
    dates = []
    file_type = 'AET' if 'AET' in directory else 'CHIRPS' if 'Chirps' in directory else 'DIFF'
    
    for file in files:
        filename = os.path.basename(file)
        date = extract_date_from_filename(filename, file_type)
        dates.append(date)
    
    dates = sorted(dates)
    
    # # Pour AET, on force la date de début à 2018-01-01
    # if file_type == 'AET':
    #     start_date = '2018-01-01'
    # else:
    #     start_date = dates[0] if dates else None
    
    start_date = dates[0] if dates else None    
    end_date = dates[-1] if dates else None
    
    # Générer toutes les dates attendues
    if start_date and end_date:
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        expected_dates = []
        current = start
        while current <= end:
            for dekad in [1, 11, 21]:  # Jours de début de chaque dekad
                dekad_date = current.replace(day=dekad)
                if start <= dekad_date <= end:
                    expected_dates.append(dekad_date.strftime('%Y-%m-%d'))
            current = (current.replace(day=1) + timedelta(days=32)).replace(day=1)
        
        missing_dates = sorted(list(set(expected_dates) - set(dates)))
        return start_date, end_date, missing_dates
    return None, None, []

################################################

# Fonction pour extraire la date à partir du nom de fichier
def extract_date_from_filename(filename: str, file_type: str) -> str:
    if file_type == 'AET':
        parts = filename.split('.')
        date_part = parts[-2]
        year, month, dekad = date_part.split('-')
        year = int(year)
        month = int(month)
        dekad = int(dekad[1])

    elif file_type == 'CHIRPS':
        parts = filename.split('.')
        year_dekad = parts[2]
        year = int(year_dekad[:2]) + 2000
        dekad = int(year_dekad[2:])
        month = (dekad - 1) // 3 + 1
        dekad = dekad % 3 if dekad % 3 != 0 else 3

    elif file_type == 'DIFF':
        parts = filename.split('-')
        year = int(parts[0])
        month = int(parts[1])
        dekad = int(parts[2][1])

    dekad_to_day = {1: 1, 2: 11, 3: 21}
    day = dekad_to_day.get(dekad, 1)
    return datetime(year, month, day).strftime('%Y-%m-%d')

# Fonction pour lire et calculer la moyenne des valeurs raster avec un masque
def read_value_with_mask_mean(file_path: str, reprojected_geometries) -> float:
    with rasterio.open(file_path) as src:
        out_image, out_transform = mask(src, reprojected_geometries, crop=True)
        data = out_image[0]
        data = np.where(data == src.nodatavals[0], np.nan, data)
        return np.nanmean(data)

# Fonction pour calculer la superficie irriguée en mètres carrés
def calculate_irrigated_area_sqm(mask_tif_path, reprojected_geometries):
    with rasterio.open(mask_tif_path) as mask_src:
        mask_cropped, _ = mask(mask_src, reprojected_geometries, crop=True)
        mask_data = mask_cropped[0]
        mask_data = np.where(mask_data == mask_src.nodatavals[0], np.nan, mask_data)
        pixel_width, pixel_height = mask_src.res
        pixel_area = pixel_width * pixel_height
        irrigated_area_sqm = np.sum(mask_data == 1) * pixel_area
    return irrigated_area_sqm

# Fonction pour calculer les moyennes sur une période spécifique
def calculate_period_average(data, start_month, period, value_column):
    val = {}
    for year in data['year'].unique():
        filtered_data = pd.DataFrame()
        for month in range(period):
            current_month = (start_month + month - 1) % 12 + 1
            current_year = year if current_month >= start_month else year + 1
            monthly_data = data[(data['year'] == current_year) & (data['month'] == current_month)]
            filtered_data = pd.concat([filtered_data, monthly_data])
        
        if not filtered_data.empty:
            val[year] = filtered_data[value_column].sum()
    
    return val

# A card component wrapper.
def ui_card(title, color, width, height, *args):
    return ui.div(
        {"class": "card mb-4", "style": f"background-color: {color}; width: {width}%; height: {height}px;"},
        ui.div(title, class_="card-header"),
        ui.div({"class": "card-body"}, *args),
    )

# Function for App 1: Download WaPOR data
# Function for App 1: Download WaPOR data
import subprocess

def download_data(start_year, end_year, start_month, end_month):
    # Loop through years and months
    for year in range(start_year, end_year + 1):
        for month in range(start_month, end_month + 1):
            month_str = f"{month:02d}"
            print(f"Downloading data for {year}-{month_str}")

            # Construct the output directory
            output_dir = os.path.join('01_aet')

            # Create the directory if it does not exist
            os.makedirs(output_dir, exist_ok=True)

            # Construct the gsutil command
            command = [
                'gsutil', '-m', 'cp',
                f'gs://fao-gismgr-wapor-3-data/DATA/WAPOR-3/MOSAICSET/L3-AETI-D/WAPOR-3.L3-AETI-D.MIT.{year}-{month_str}-D1.tif',
                f'gs://fao-gismgr-wapor-3-data/DATA/WAPOR-3/MOSAICSET/L3-AETI-D/WAPOR-3.L3-AETI-D.MIT.{year}-{month_str}-D2.tif',
                f'gs://fao-gismgr-wapor-3-data/DATA/WAPOR-3/MOSAICSET/L3-AETI-D/WAPOR-3.L3-AETI-D.MIT.{year}-{month_str}-D3.tif',
                output_dir
            ]

            try:
                print('Executing command:', ' '.join(command))
                subprocess.run(command, check=True, shell=(os.name == 'nt'))  # Use shell=True for Windows
                print(f"Successfully downloaded data for {year}-{month_str}")
            except subprocess.CalledProcessError as e:
                print(f"Error downloading data for {year}-{month_str}: {e}")


# Function for App 2: Download and extract CHIRPS data
def download_and_extract_data(start_year, end_year, start_month, end_month):
    import requests
    from tqdm import tqdm
    import gzip
    import shutil

    def download_file(url, destination_path):
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))
            with open(destination_path, 'wb') as f, tqdm(
                desc=destination_path,
                total=total_size,
                unit='iB',
                unit_scale=True
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    pbar.update(size)
            return True
        else:
            print(f"Failed to download {url}. Status code: {response.status_code}")
            return False

    base_url = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_dekad/tifs/"
    years = [str(year) for year in range(start_year, end_year + 1)]
    months = [f"{month:02d}" for month in range(start_month, end_month + 1)]
    
    destination = os.path.abspath("01_chirps")
    os.makedirs(destination, exist_ok=True)

    for year in years:
        for month in months:
            for dekad in range(1, 4):  # CHIRPS has 3 dekads per month
                filename = f"chirps-v2.0.{year}.{month}.{dekad}.tif.gz"
                url = f"{base_url}{filename}"
                file_path = os.path.join(destination, filename)
                
                try:
                    print(f"Downloading {filename}...")
                    if download_file(url, file_path):
                        print(f"Extracting {filename}...")
                        with gzip.open(file_path, 'rb') as f_in:
                            with open(file_path[:-3], 'wb') as f_out:  # Remove '.gz' to get the .tif filename
                                shutil.copyfileobj(f_in, f_out)
                        os.remove(file_path)  # Remove the .gz file after extraction
                    else:
                        print(f"Skipping extraction of {filename} due to download failure")
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
                    raise



######################## Start 01_aet downscaling ############################# ##########################################

output_file_aet_01 = 'downscaling/tmp1/reproj_'
input_file_aet_01 = 'downscaling/tmp1/reproj_cropped_'
resampled_file_aet_01 = "downscaling/tmp1/resampled_output_"
my_shpfile_aet_01 = "downscaling/shp/SNIC_30000_V2.shp"
mask_file_path_aet_01 = "downscaling/shp/SNIC_30000_V2.tif"
num_cores = 4

# Function to update GeoTIFF with a mask
def update_tif_with_mask_aet_01(tif_file_path, mask_file_path_aet_01, output_file_aet_01_path, mask_value=-1):
    with rasterio.open(tif_file_path, 'r') as tif_dataset:
        tif_data = tif_dataset.read(1)
        nodata_value = tif_dataset.profile['nodata']
        tif_data[tif_data < -1] = nodata_value

        with rasterio.open(mask_file_path_aet_01, 'r') as mask_dataset:
            mask_data = mask_dataset.read(1)
            if tif_dataset.bounds == mask_dataset.bounds and \
               tif_dataset.crs == mask_dataset.crs and \
               tif_dataset.width == mask_dataset.width and \
               tif_dataset.height == mask_dataset.height:
                tif_data[mask_data == mask_value] = nodata_value
                with rasterio.open(output_file_aet_01_path, 'w', **tif_dataset.profile) as output_dataset:
                    output_dataset.write(tif_data, 1)
                print("Operation completed successfully.")
            else:
                print("Error: The extent, projection, or pixel numbers of the two datasets do not match.")

# Function to perform average segmentation based on shapefile
def avg_seg_aet_01(filein, fileout):
    with rasterio.open(filein) as src:
        transform = src.transform
        dtype = src.dtypes[0]
        nodata = -9999
        shapefile = gpd.read_file(my_shpfile_aet_01)
        shapefile = shapefile.to_crs(src.crs)
        shapefile_bounds = shapefile.total_bounds
        shapefile_polygon = box(*shapefile_bounds)
        raster_cropped, transform_cropped = mask(src, [shapefile_polygon], crop=True)
        height_cropped, width_cropped = raster_cropped.shape[1:]
        raster_data_cropped = raster_cropped[0]
        raster_data_cropped[raster_data_cropped < -1] = nodata
        output_data = np.full((height_cropped, width_cropped), nodata, dtype=dtype)

        for idx, feature in shapefile.iterrows():
            geometry = feature.geometry
            feature_mask = rasterio.features.geometry_mask([geometry], out_shape=(height_cropped, width_cropped),
                                                           transform=transform_cropped)
            masked_values = np.ma.array(raster_data_cropped, mask=~feature_mask, fill_value=nodata)
            if np.any(masked_values.mask == True):
                average_value = np.nanmean(masked_values.data[masked_values.mask == True])
            else:
                average_value = nodata
            output_data[masked_values.mask == True] = average_value

        with rasterio.open(fileout, 'w', driver='GTiff', height=height_cropped,
                           width=width_cropped, count=1, dtype=dtype, nodata=nodata, crs=src.crs,
                           transform=transform_cropped) as dst:
            dst.write(output_data, 1)

# Main processing function for each dekad
def process_nmfic_aet_01(nmfic):
    year, month, dekad = nmfic[:4], nmfic[5:7], nmfic[-2:]
    output_file_path = f"outputs/AET_dekad_brut_m/WAPOR-3.L3-AETI-D.MIT.{year}-{month}-{dekad}.tif"
    
    # Vérifier si le fichier existe déjà
    if os.path.exists(output_file_path):
        print(f"Le fichier {output_file_path} existe déjà. Downscaling ignoré.")
        return True  # Indique que le fichier existe déjà
        
    try:
        input_raw_file_aet_01_path = f"01_aet/WAPOR-3.L3-AETI-D.MIT.{year}-{month}-{dekad}.tif"
        final_file_aet_01_path = f"downscaling/tmp1/WAPOR-3.L3-AETI-D.MIT.{year}-{month}-{dekad}.tif"

        # Vérifier si le fichier d'entrée existe
        if not os.path.exists(input_raw_file_aet_01_path):
            raise FileNotFoundError(f"Le fichier d'entrée n'existe pas: {input_raw_file_aet_01_path}")

        print(f'Traitement de: {input_raw_file_aet_01_path}')
        
        # Reprojection
        src_srs = osr.SpatialReference()
        src_srs.ImportFromEPSG(4326)
        target_srs = osr.SpatialReference()
        target_srs.ImportFromEPSG(32631)  # UTM Zone 31N pour l'Algérie nord-ouest

        input_ds = gdal.Open(input_raw_file_aet_01_path)
        if input_ds is None:
            raise RuntimeError(f"Impossible d'ouvrir le fichier: {input_raw_file_aet_01_path}")

        # Créer les répertoires de sortie s'ils n'existent pas
        os.makedirs(os.path.dirname(final_file_aet_01_path), exist_ok=True)
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

        output_file_aet_011 = f"{output_file_aet_01}{nmfic}.tif"
        gdal.Warp(output_file_aet_011, input_ds, format='GTiff', dstSRS=target_srs)
        input_ds = None

        shapefile_ds = ogr.Open(my_shpfile_aet_01)
        if shapefile_ds is None:
            raise RuntimeError(f"Impossible d'ouvrir le shapefile: {my_shpfile_aet_01}")

        layer = shapefile_ds.GetLayer()
        extent = layer.GetExtent()

        output_raster = f"{input_file_aet_01}{nmfic}.tif"
        gdal.Warp(output_raster, output_file_aet_011, format="GTiff", 
                 outputBounds=(extent[0], extent[2], extent[1], extent[3]))
        shapefile_ds = None

        snic_ds = gdal.Open(my_shpfile_aet_01.replace(".shp", ".tif"))
        if snic_ds is None:
            raise RuntimeError("Impossible d'ouvrir le fichier SNIC")

        target_resolution = snic_ds.GetGeoTransform()[1]
        snic_ds = None

        output_raster = gdal.Open(output_raster, gdal.GA_Update)
        resampled_file_aet_011 = f"{resampled_file_aet_01}{nmfic}.tif"

        # Resample the cropped raster
        gdal.Warp(resampled_file_aet_011, output_raster, format="GTiff", 
                 outputBounds=(extent[0], extent[2], extent[1], extent[3]),
                 xRes=target_resolution, yRes=target_resolution, 
                 creationOptions=["TILED=YES", "COMPRESS=LZW"])

        # Apply average segmentation
        avg_seg_aet_01(resampled_file_aet_011, final_file_aet_01_path)

        # Apply mask to the final output
        update_tif_with_mask_aet_01(final_file_aet_01_path, mask_file_path_aet_01, output_file_path)

        # Clean up temporary files
        output_raster = None
        if os.path.exists(output_file_aet_011):
            os.remove(output_file_aet_011)
        if os.path.exists(resampled_file_aet_011):
            os.remove(resampled_file_aet_011)

        print(f"Traitement terminé avec succès pour {nmfic}")
        return True

    except Exception as e:
        print(f"Erreur lors du traitement de {nmfic}: {str(e)}")
        raise

######################## End 01_aet downscaling #############################

######################Start 01_chirps downscaling ##########################

# Variables globales pour les fichiers
output_file_01_chirps = 'downscaling/tmp2/reproj_'
input_file_01_chirps = 'downscaling/tmp2/reproj_cropped_'
resampled_file_01_chirps = "downscaling/tmp2/resampled_output_"
my_shpfile_01_chirps = "downscaling/shp/SNIC_30000_V2.shp"
mask_file_path_01_chirps = "downscaling/shp/SNIC_30000_V2.tif"


# Fonction pour le traitement des fichiers CHIRPS
def process_nmfic_01_chirps(year, month, dekad):
    # Convertir le mois en entier
    month = int(month)
    year = int(year)

    # Format du mois sur 2 chiffres
    month_str = f"{month:02d}"
    
    # Déterminer le numéro de dekad
    if isinstance(dekad, str) and dekad.startswith('D'):
        dekad_num = int(dekad[1])
    else:
        dekad_num = int(dekad)

    # Vérifier que le dekad est valide
    if dekad_num not in [1, 2, 3]:
        print(f"Dekad invalide: {dekad_num}")
        return

    # Calcul du numéro de dekad global
    dekad_global = (month - 1) * 3 + dekad_num
    
    # Construire le chemin du fichier de sortie
    output_file_path = f"outputs/Chirps_dekad_brut_m/chirps-v2.0.{year-2000}{dekad_global:02d}.tif"
    
    # Vérifier si le fichier existe déjà
    if os.path.exists(output_file_path):
        print(f"Le fichier {output_file_path} existe déjà. Downscaling ignoré.")
        return True  # Indique que le fichier existe déjà

    # Nom du fichier d'entrée (format CHIRPS)
    input_raw_file = f"01_chirps/chirps-v2.0.{year}.{month_str}.{dekad_num}.tif"
    
    # Vérifier si le fichier d'entrée existe
    if not os.path.exists(input_raw_file):
        print(f"Error: The input file does not exist: {input_raw_file}")
        return
    
    print(f"Processing the dekade : {year} {month} {dekad}")
    
    # Vérifier l'existence du fichier d'entrée
    if not os.path.exists(input_raw_file):
        raise FileNotFoundError(f"Le fichier d'entrée n'existe pas: {input_raw_file}")

    print(f'Processing: {input_raw_file}')

    # Reprojection
    src_srs = osr.SpatialReference()
    src_srs.ImportFromEPSG(4326)
    target_srs = osr.SpatialReference()
    target_srs.ImportFromEPSG(32631)  # UTM Zone 31N pour l'Algérie nord-ouest

    try:
        input_ds = gdal.Open(input_raw_file)
        if input_ds is not None:
            # Première reprojection
            output_file_01_chirps = f"downscaling/tmp2/reproj_{year}{dekad_num}.tif"
            gdal.Warp(output_file_01_chirps, input_ds, format='GTiff', dstSRS=target_srs)
            input_ds = None

            # Charger le shapefile pour découper l'image
            shapefile_ds = ogr.Open(my_shpfile_01_chirps)
            if shapefile_ds is None:
                print(f"Failed to open shapefile: {my_shpfile_01_chirps}")
                return
                
            layer = shapefile_ds.GetLayer()
            extent = layer.GetExtent()

            # Obtenir la résolution cible à partir du fichier de masque
            with rasterio.open(mask_file_path_01_chirps) as mask_ds:
                target_transform = mask_ds.transform
                target_width = mask_ds.width
                target_height = mask_ds.height
                target_crs = mask_ds.crs

            # Créer le fichier raster avec les mêmes dimensions que le masque
            output_raster = f"{input_file_01_chirps}{year}{month_str}{dekad_num}.tif"
            gdal.Warp(output_raster, 
                    output_file_01_chirps, 
                    format="GTiff",
                    outputBounds=(extent[0], extent[2], extent[1], extent[3]),
                    width=target_width,
                    height=target_height,
                    dstSRS=target_crs)
            
            shapefile_ds = None

            # Appliquer la segmentation moyenne
            avg_seg_01_chirps(output_raster, f"downscaling/tmp2/AVG_50k_{year}{dekad_global:02d}.tif")

            # Appliquer le masque au fichier final
            update_tif_with_mask_01_chirps(f"downscaling/tmp2/AVG_50k_{year}{dekad_global:02d}.tif", mask_file_path_01_chirps, output_file_path)

            # Nettoyer les fichiers temporaires
            if os.path.exists(output_file_01_chirps):
                os.remove(output_file_01_chirps)
            if os.path.exists(output_raster):
                os.remove(output_raster)
                
        else:
            print(f"Failed to open the input file: {input_raw_file}")
            
    except Exception as e:
        print(f"Error processing {input_raw_file}: {str(e)}")


# Fonction pour appliquer un masque sur le GeoTIFF
def update_tif_with_mask_01_chirps(tif_file_path, mask_file_path_01_chirps, output_file_01_chirps_path, mask_value=-1):
    with rasterio.open(tif_file_path, 'r') as tif_dataset:
        tif_data = tif_dataset.read(1)
        nodata_value = tif_dataset.profile['nodata']
        tif_data[tif_data < -1] = nodata_value

        with rasterio.open(mask_file_path_01_chirps, 'r') as mask_dataset:
            mask_data = mask_dataset.read(1)
            if tif_dataset.bounds == mask_dataset.bounds and \
               tif_dataset.crs == mask_dataset.crs and \
               tif_dataset.width == mask_dataset.width and \
               tif_dataset.height == mask_dataset.height:
                tif_data[mask_data == mask_value] = nodata_value
                with rasterio.open(output_file_01_chirps_path, 'w', **tif_dataset.profile) as output_dataset:
                    output_dataset.write(tif_data, 1)
                print("Operation completed successfully.")
            else:
                print("Error: The extent, projection, or pixel numbers of the two datasets do not match.")


# Fonction pour la segmentation moyenne (simplifiée)
def avg_seg_01_chirps(filein, fileout):
    with rasterio.open(filein) as src:
        transform = src.transform
        dtype = src.dtypes[0]
        nodata = -9999
        shapefile = gpd.read_file(my_shpfile_01_chirps)
        shapefile = shapefile.to_crs(src.crs)
        shapefile_bounds = shapefile.total_bounds
        shapefile_polygon = box(*shapefile_bounds)
        raster_cropped, transform_cropped = mask(src, [shapefile_polygon], crop=True)
        height_cropped, width_cropped = raster_cropped.shape[1:]
        raster_data_cropped = raster_cropped[0]
        raster_data_cropped[raster_data_cropped < -1] = nodata
        output_data = np.full((height_cropped, width_cropped), nodata, dtype=dtype)

        for idx, feature in shapefile.iterrows():
            geometry = feature.geometry
            feature_mask = rasterio.features.geometry_mask([geometry], out_shape=(height_cropped, width_cropped),
                                                           transform=transform_cropped)
            masked_values = np.ma.array(raster_data_cropped, mask=~feature_mask, fill_value=nodata)
            average_value = np.nanmean(masked_values.data[masked_values.mask == True]) if np.any(masked_values.mask == True) else nodata
            output_data[masked_values.mask == True] = average_value

        with rasterio.open(fileout, 'w', driver='GTiff', height=height_cropped, width=width_cropped, count=1, dtype=dtype,
                           nodata=nodata, crs=src.crs, transform=transform_cropped) as dst:
            dst.write(output_data, 1)


############################ End 01_chirps_downscaling ####################################################

######################## Start Difference #####################################


def get_rainfall_filename(year, month, dekade):
    dekade_number = (month - 1) * 3 + dekade
    # return f"AVG_50k_{year}{str(dekade_number).zfill(2)}.tif"
    return f"chirps-v2.0.{year-2000}{str(dekade_number).zfill(2)}.tif"

def get_et_filename(year, month, dekade):
    return f"WAPOR-3.L3-AETI-D.MIT.{year}-{str(month).zfill(2)}-D{dekade}.tif"

def read_tif_values(tif_file, geometry):
    with rasterio.open(tif_file) as src:
        mask = geometry_mask([geometry], transform=src.transform, invert=True, out_shape=(src.height, src.width))
        values = src.read(1)[mask]
    return values

def calculate_and_write_difference(et_file, rainfall_file, output_file, gdf):
    # Vérifier si le fichier de sortie existe déjà
    if os.path.exists(output_file):
        print(f"Le fichier {output_file} existe déjà. Calcul de différence ignoré.")
        return True  # Indique que le fichier existe déjà

    try:
        with rasterio.open(et_file) as src_et:
            et_meta = src_et.meta
            et_data = np.zeros_like(src_et.read(1))

        with rasterio.open(rainfall_file) as src_rf:
            rf_data = src_rf.read(1)
        
        for idx, row in gdf.iterrows():
            if (idx + 1) % 1000 == 0:
                print(f"Processing segment {idx + 1} of {len(gdf)}")
            geometry = row['geometry']

            et_values = read_tif_values(et_file, geometry)
            rf_values = read_tif_values(rainfall_file, geometry)

            if len(et_values) > 0 and len(rf_values) > 0:
                difference = et_values[0] - rf_values[0]
                if difference < 0.:
                    difference = 0.
            else:
                difference = np.nan

            mask = geometry_mask([geometry], transform=src_et.transform, invert=True, out_shape=et_data.shape)
            et_data[mask] = difference

        et_meta.update(dtype=rasterio.float32)
        with rasterio.open(output_file, 'w', **et_meta) as dst:
            dst.write(et_data.astype(rasterio.float32), 1)
            
        return True
    except Exception as e:
        print(f"Erreur lors du calcul de la différence: {str(e)}")
        return False

def update_tif_with_mask(tif_file_path, mask_file_path, output_file_path, mask_value=-1):
    with rasterio.open(tif_file_path, 'r') as tif_dataset:
        tif_data = tif_dataset.read(1)
        nodata_value = tif_dataset.profile['nodata']
        tif_data[tif_data < -1] = nodata_value
        
        with rasterio.open(mask_file_path, 'r') as mask_dataset:
            mask_data = mask_dataset.read(1)
            if tif_dataset.bounds == mask_dataset.bounds and \
               tif_dataset.crs == mask_dataset.crs and \
               tif_dataset.width == mask_dataset.width and \
               tif_dataset.height == mask_dataset.height:
                tif_data[mask_data == mask_value] = nodata_value
                with rasterio.open(output_file_path, 'w', **tif_dataset.profile) as output_dataset:
                    output_dataset.write(tif_data, 1)
                print("Operation completed successfully.")
            else:
                print("Error: The extent, projection, or pixel numbers of the two datasets do not match.")


def process_diff_data(start_year, start_month, start_dekade, end_year, end_month, end_dekade, gdf, model_path, status_messager=None):
    et_directory = "outputs/AET_dekad_brut_m/"
    rainfall_directory = "outputs/Chirps_dekad_brut_m/"
    output_directory = "outputs/difirence_dekad_brut_m/"

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    current_year = start_year
    current_month = start_month
    current_dekade = start_dekade

    # Calculer le nombre total de décades à traiter
    total_dekades = 0
    temp_year, temp_month, temp_dekade = start_year, start_month, start_dekade
    while not (temp_year == end_year and temp_month == end_month and temp_dekade == end_dekade):
        total_dekades += 1
        temp_dekade += 1
        if temp_dekade > 3:
            temp_dekade = 1
            temp_month += 1
            if temp_month > 12:
                temp_month = 1
                temp_year += 1
    total_dekades += 1  # Inclure la dernière décade

    with tqdm(total=total_dekades, desc="Processing Differences") as pbar:
        while True:
            et_file = os.path.join(et_directory, get_et_filename(current_year, current_month, current_dekade))
            rainfall_file = os.path.join(rainfall_directory, get_rainfall_filename(current_year, current_month, current_dekade))
            
            current_status = f'Processing dekade: {current_year}-{str(current_month).zfill(2)}-D{current_dekade}'
            if status_messager:
                status_messager(current_status)
            pbar.set_description(current_status)
            
            if os.path.exists(et_file) and os.path.exists(rainfall_file):
                output_file = os.path.join(output_directory, f"{current_year}-{str(current_month).zfill(2)}-D{current_dekade}.tif")
                output_final_file = os.path.join(output_directory, f"{current_year}-{str(current_month).zfill(2)}-D{current_dekade}_masked.tif")
                
                if status_messager:
                    status_messager(f"Calculating difference for {current_year}-{str(current_month).zfill(2)}-D{current_dekade}")
                calculate_and_write_difference(et_file, rainfall_file, output_file, gdf)
                
                if status_messager:
                    status_messager(f"Applying mask to {output_file}")
                update_tif_with_mask(output_file, model_path, output_final_file)
                
                if os.path.exists(output_file):
                    os.remove(output_file)
                    os.rename(output_final_file, output_file)
                    if status_messager:
                        status_messager(f"Successfully processed: {output_file}")
            else:
                missing_files = []
                if not os.path.exists(et_file):
                    missing_files.append("ET file")
                if not os.path.exists(rainfall_file):
                    missing_files.append("Rainfall file")
                message = f"Missing {' and '.join(missing_files)} for {current_year}-{current_month}-D{current_dekade}"
                if status_messager:
                    status_messager(message)
                print(message)

            pbar.update(1)

            current_dekade += 1
            if current_dekade > 3:
                current_dekade = 1
                current_month += 1
                if current_month > 12:
                    current_month = 1
                    current_year += 1

            if current_year == end_year and current_month == end_month and current_dekade == end_dekade:
                if status_messager:
                    status_messager("Processing completed successfully!")
                break


###################### End Difference #############################################################


# Interface utilisateur
app_ui = ui.page_navbar(
    ui.nav_spacer(),
    ui.nav_panel("Map Tools",
        ui_card(
                "Objective", "oldlace", "100%", 130,
                "The main objective of this Water Use Tool Dashboard is to monitor rainfall patterns, open field crop evapotranspiration, and water use in the basin. This helps decision-makers understand seasonal fluctuations and refine water allocation for improved management. Note: the irrigation water use data does not distinguish between surface and groundwater pumping.",
                ui.download_button("download_manual", "Download User Manual")
                ),
        ui.layout_sidebar(
            ui.panel_sidebar(
                ui.input_file("geojson_file", "Upload GeoJSON File", accept=[".geojson"]),
                ui.value_box(
                    "Area Irrigated Hec",
                    ui.output_text("area_irrigated"),
                    theme="gradient-blue-indigo",
                    showcase=icon_svg("globe"),
                ),
                ui_card(
                    "Total Water Use (m3)", "oldlace", "100%", 400,
                    output_widget("results_plot_ws", height="400px"),
                    ui.download_button("download_roi_data", "Download ROI Data (CSV)"),
                ),
            ),
            ui.panel_main(
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Map"),
                        ui.output_ui("map"),
                        class_="mb-6",
                    ),
                ),
            ),
        ),
    ),
    ui.nav_panel("Data Analysis",
        ui.layout_columns(
            ui.card(
                ui.card_header("Precipitation (mm)"),
                output_widget("box_season_precip", height="320"),
                class_="mb-3",
            ),
            ui.card(
                ui.card_header("Evapotranspiration (mm)"),
                output_widget("box_season_aet", height="320"), 
                class_="mb-3",
            ),
            ui.card(
                ui.card_header("Water use (m3)"),
                output_widget("box_season_ws", height="320"),
                class_="mb-3",
            ),
        ),
        ui.layout_columns(
            ui.card(
                ui.card_header("Precipitation (mm)"),
                output_widget("bar_month_year_precip", height="320"),
                class_="mb-3",
            ),
            ui.card(
                ui.card_header("Evapotranspiration (mm)"),
                output_widget("bar_month_year_aet", height="320"),
                class_="mb-3",
            ),
            ui.card(
                ui.card_header("Water use (m3)"),
                output_widget("bar_month_year_ws", height="320"),
                class_="mb-3",
            ),
        ),
        ui.layout_columns(
            ui.input_slider("start_month", "Select start month:", min=1, max=12, value=1, step=1),
            ui.input_slider("period", "Select period:", min=1, max=12, value=12, step=1),
            ui.card(
                ui.card_header("Precipitation (mm)"),
                output_widget("yearly_precip", height="400px"),
                class_="mb-2",
            ),
            ui.card(
                ui.card_header("Evapotranspiration (mm)"),
                output_widget("yearly_evap", height="400px"),
                class_="mb-2",
            ),

            ui.card(
                ui.card_header("Total Water Use (m3)"),
                output_widget("yearly_ws", height="400px"),
                class_="mb-2",
            ),
        ),
        ui.layout_columns(
            ui.card(
                ui.card_header("Precipitation & AET Monthly Evolution"),
                output_widget("results_plot", height="400px"),
                class_="mb-2",
            ),

            ui.card(
                ui.card_header("Downoad ROI Data"),
                ui.download_button("download_csv", "Download CSV"),
                ui.output_table("merged_data_table"),
                class_="mb-2", 
            ),
            
        ),
        ui.layout_columns(
            ui.card(
                ui.card_header("Analyse des données manquantes par répertoire"),
                ui.output_table("directory_analysis"),
                class_="mb-2",
            ),
        ),        
    ),
    ui.nav_panel("Preprocessing",
                 ui.navset_tab(
                    ui.nav(
                        "Preprocessing",
                        ui.layout_columns(
                            
                            ui_card("Tools to automatically Download WaPOR Data ", "oldlace", "20%", 700, 
                                
                                ui.input_select("wapor_start_year", "Start year", [str(i) for i in range(2018, 2025)]),
                                ui.input_select("wapor_end_year", "End year", [str(i) for i in range(2018, 2025)]),
                                ui.input_select("wapor_start_month", "Month de début", [f"{i:02d}" for i in range(1, 13)]),
                                ui.input_select("wapor_end_month", "Month de fin", [f"{i:02d}" for i in range(1, 13)]),
                                ui.input_action_button("wapor_download", "Download Wapor Data"),
                                ui.output_text_verbatim("wapor_status"),
                                
                                ),
                            ui_card("Tools to automatically Download CHIRPS Data","oldlace", "20%", 700, 
                                    
                                    ui.input_select("chirps_start_year", "Start year", [str(i) for i in range(2018, 2025)]),
                                    ui.input_select("chirps_end_year", "End year", [str(i) for i in range(2018, 2025)]),
                                    ui.input_select("chirps_start_month", "Start month", [f"{i:02d}" for i in range(1, 13)]),
                                    ui.input_select("chirps_end_month", "End month", [f"{i:02d}" for i in range(1, 13)]),
                                    ui.input_action_button("chirps_download", "Download Chirps Data"),
                                    ui.output_text_verbatim("chirps_status"),
                                    
                                    ),
                            ui_card("ETa Downscaling", "oldlace", "20%", 700, 
                                    
                                    
                                    ui.input_select("year_aet_01", "Year", choices=[str(y) for y in range(2018, 2025)], selected="2018"),
                                    ui.input_select("month_aet_01", "Month", choices=[f"{m:02}" for m in range(1, 13)], selected="01"),
                                    ui.input_select("dekad_aet_01", "Dekad", choices=["D1", "D2", "D3"], selected="D1"),
                                    ui.input_action_button("start_process_aet_01", "Start Processing"),
                                    ui.output_text_verbatim("output_status_aet_01")
                                    
                                    
                                    ),
                            ui_card("Chirps Downscaling", "oldlace", "20%", 700, 
                                
                                    
                                    ui.input_select("year_01_chirps", "Year", choices=[str(y) for y in range(2018, 2025)], selected="2018"),
                                    ui.input_select("month_01_chirps", "Month", choices=[f"{m:02}" for m in range(1, 13)], selected="01"),
                                    ui.input_select("dekad_01_chirps", "Décade", choices=["D1", "D2", "D3"], selected="D1"),
                                    ui.input_action_button("start_process_01_chirps", "Start Processing"),
                                    ui.output_text_verbatim("output_status_01_chirps")
                                    
                                    ),   

                            ui_card("Difference Processing Tool", "oldlace", "20%", 700, 
                                
                                   
                                    ui.input_numeric("year_diff", "Year", value=2024, min=2000, max=2100),
                                    ui.input_numeric("month_diff", "Month", value=1, min=1, max=12),
                                    ui.input_numeric("dekad_diff", "Dekad", value=1, min=1, max=3),
                                    ui.input_action_button("run_diff", "Calculate Difference"),
                                    ui.output_text("output_status_diff")
                                    
                                    
                                    ),                                   
                                
                            
                        ),

                        

                        
                    )
                    
    )
        
    ),

    ui.nav_panel("À propos",
        ui.panel_main(
            ui.h2("À propos de l'outil d'utilisation de l'eau en Algérie"),
            ui.markdown("""
            ### Équipe
            **Développeurs IWMI**: Karim Bergaoui, Makram Belhaj Fraj, et Hatem Cherif  
            **Superviseurs IWMI**: Petra Schmitter et Moctar Dembélé  

            ### Groupe de travail Algérie
            Cellule de Numérisation WaPOR, Unité Numérique, Département des Statistiques et des Stratégies, Ministère de l'Agriculture, Algérie:
            - M. Ahmed BELLAHRECHE (Ingénieur Agronome Senior à l'INRAA supervisant la CNW: a.bellahreche@gmail.com)
            - M. Djemai AKSA (Ingénieur Informaticien au MADR, d.aksa@madr.gov.dz)
            - Mme. Nadia HADDAD (Spécialiste SIG au MADR, nadia.hadd@gmail.com)
            - Mme. Hiba TOUIHAT (Ingénieur de Développement au MADR, ayatimo25@gmail.com)
            - Mme. Selsabil CHOUIHAT (Agroéconomiste au MADR, bila-sali@hotmail.fr)
            - Mme. Nacera HADJOUT (Hydrologue Senior, Coordinatrice de Projet à AGIRE, Ministère de l'Hydraulique, hadjoutnacera.agire@gmail.com)

            **Agent de Liaison Senior**: M. Farouk Belkhir au nom du Président du COPIL de la partie algérienne, M. Ali Ferrah, DG INRAA.

            ### Facilitation
            **Facilité par**: Riad Mezaouar et Imen Farrah (FAO Algérie)

            ### Membres du COPIL
            (À communiquer par la FAO)

            **Présidents du COPIL**: Mme. Irina Buttoud (Représentante de la FAO en Algérie) et Ali Ferrah (DG INRAA)
            """)
        )          
    ),
      
    title=[
        ui.img(src="IWMI1.png", class_="navbar-logo"),
        ui.img(src="fao.png", class_="navbar-logo"),
        ui.img(src="wapor.png", class_="navbar-logo"),
        "Dynamic Dashboard for the Water Use Tool - Mitidja Ouest, Algeria (Statistics Department)"
    ],
    footer=ui.tags.style(
        """
        .navbar-logo {
            height: 100%;
            max-height: 50px;
            margin-right: 10px;
        }
        .navbar-header {
            display: flex;
            align-items: center;
        }
        """
    ),
    id="navbar_id",
    bg="beige",
    inverse=False,
)


def server(input, output, session):
    # Status messagers for processing
    status_messager_aet_01 = reactive.Value("Waiting to start...")
    status_messager_01_chirps = reactive.Value("Waiting to start...")
    status_messager_diff = reactive.Value("Waiting to start...")
    
    @session.download(filename="manual.pdf")
    def download_manual():
        with open("www/methodology.pdf", "rb") as f:
            yield f.read()

    @session.download(filename="roi_data.csv")
    def download_roi_data():
        try:
            data = process_data()
            if data is not None and not data.empty:
                # Préparer les données pour l'export
                export_data = data.copy()
                export_data = export_data.reset_index()
                
                # Sélectionner et renommer les colonnes
                columns_to_export = {
                    'Date': 'Date',
                    'AET': 'Evapotranspiration (mm)',
                    'Rainfall': 'Precipitation (mm)',
                    'Total_Water_Cubic_Meters': 'Water Use (m³)',
                    'Irrigated_Area_Hec': 'Irrigated Area (ha)',
                    'Season': 'Season',
                    'year': 'Year',
                    'month': 'Month'
                }
                
                export_data = export_data[list(columns_to_export.keys())]
                export_data.columns = list(columns_to_export.values())
                
                # Utiliser StringIO pour créer le CSV
                output = StringIO()
                export_data.to_csv(output, index=False)
                return output.getvalue()
            return "No data available"
        except Exception as e:
            print(f"Error in download_roi_data: {str(e)}")
            return "Error generating CSV file"

    # Download tracking
    download_progress = reactive.Value(0)
    download_status_text = reactive.Value("")
    total_files_processed = reactive.Value(0)
    total_files = reactive.Value(0)

    @output
    @render.text
    def download_progress_text():
        if total_files.get() > 0:
            return f"Progress: {total_files_processed.get()}/{total_files.get()} files ({(total_files_processed.get() / total_files.get() * 100):.1f}%)"
        return ""

    @output
    @render.text
    def download_status():
        return download_status_text.get()

    # Status outputs
    @output
    @render.text
    def output_status_aet_01():
        return status_messager_aet_01.get()

    @output
    @render.text
    def output_status_01_chirps():
        return status_messager_01_chirps.get()

    @output
    @render.text
    def output_status_diff():
        return status_messager_diff.get()

    # Data processing functions
    @reactive.Calc
    def process_data():
        file_info = input.geojson_file()

        if not file_info:
            geojson_path = path_full_basin
        else:
            geojson_path = file_info[0].get('datapath')

        if not geojson_path or not os.path.exists(geojson_path):
            print("GeoJSON file path is None or does not exist.")
            return None

        with open(geojson_path, 'r') as f:
            geojson_data = json.load(f)
            if not geojson_data or 'features' not in geojson_data:
                print("Invalid GeoJSON data.")
                return None

        transformer = Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=True)
        reprojected_geometries = [shapely.ops.transform(transformer.transform, shape(feature["geometry"])) for feature in geojson_data["features"]]

        # Process AET data
        aet_timeseries = []
        for filename in os.listdir(aet_dir):
            if filename.endswith(".tif"):
                try:
                    date_str = extract_date_from_filename(filename, 'AET')
                    aet_value = read_value_with_mask_mean(os.path.join(aet_dir, filename), reprojected_geometries)
                    aet_timeseries.append((date_str, aet_value))
                except Exception as e:
                    print(f"Error processing AET file {filename}: {e}")

        aet_df = pd.DataFrame(aet_timeseries, columns=["Date", "AET"])
        aet_df["Date"] = pd.to_datetime(aet_df["Date"], errors='coerce')
        aet_df = aet_df.dropna(subset=["Date"])
        aet_df.set_index("Date", inplace=True)
        aet_df = aet_df.sort_index()

        # Process CHIRPS data
        chirps_timeseries = []
        for filename in os.listdir(chirps_dir):
            if filename.endswith(".tif"):
                try:
                    date_str = extract_date_from_filename(filename, 'CHIRPS')
                    rainfall_value = read_value_with_mask_mean(os.path.join(chirps_dir, filename), reprojected_geometries)
                    chirps_timeseries.append((date_str, rainfall_value))
                except Exception as e:
                    print(f"Error processing CHIRPS file {filename}: {e}")

        chirps_df = pd.DataFrame(chirps_timeseries, columns=["Date", "Rainfall"])
        chirps_df["Date"] = pd.to_datetime(chirps_df["Date"], errors='coerce')
        chirps_df = chirps_df.dropna(subset=["Date"])
        chirps_df.set_index("Date", inplace=True)
        chirps_df = chirps_df.sort_index()

        # Process difference data
        diff_timeseries = []
        for filename in os.listdir(diff_dir):
            if filename.endswith(".tif"):
                try:
                    date_str = extract_date_from_filename(filename, 'DIFF')
                    diff_value = read_value_with_mask_mean(os.path.join(diff_dir, filename), reprojected_geometries)
                    diff_timeseries.append((date_str, diff_value))
                except Exception as e:
                    print(f"Error processing difference file {filename}: {e}")

        diff_df = pd.DataFrame(diff_timeseries, columns=["Date", "Difference"])
        diff_df["Date"] = pd.to_datetime(diff_df["Date"], errors='coerce')
        diff_df = diff_df.dropna(subset=["Date"])
        diff_df.set_index("Date", inplace=True)
        diff_df = diff_df.sort_index()

        # Merge all dataframes
        merged_df = pd.merge(aet_df, chirps_df, left_index=True, right_index=True, how="outer")
        merged_df = pd.merge(merged_df, diff_df, left_index=True, right_index=True, how="outer")

        # Calculate additional metrics
        irrigated_area_sqm = calculate_irrigated_area_sqm(mask_tif_path, reprojected_geometries)
        irrigated_area_hectares = irrigated_area_sqm / 10000

        merged_df['Total_Water_Cubic_Meters'] = merged_df['Difference'] * irrigated_area_sqm / 1000
        merged_df['Irrigated_Area_Hec'] = irrigated_area_hectares
        merged_df['Season'] = merged_df.index.month.map(get_season)
        merged_df['year'] = merged_df.index.year
        merged_df['month'] = merged_df.index.month

        return merged_df

    # Map related functions
    @reactive.Calc
    def parsed_file():
        geojson_file = input.geojson_file()
        if not geojson_file:
            return gpd.read_file("geojson/4.geojson")
        return gpd.read_file(geojson_file[0]["datapath"])
    
    @reactive.Calc
    def generate_map():
        gdf = parsed_file()
        if gdf.empty:
            gdf = gpd.read_file("geojson/4.geojson")

        bounds = gdf.total_bounds
        center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]

        m = leafmap.Map(location=center, zoom_start=12)
        m.add_gdf(gdf)
        m.add_gdf(bassin, layer_name="My Shapefile Layer")
        m.add_tile_layer(
            url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            name="Esri Satellite",
            attribution="Esri",
        )
        Draw(export=True).add_to(m)
        return m

    @output
    @render.ui
    def map():
        map_widget = generate_map()
        if map_widget is None:
            return ui.tags.div("No GeoJSON file uploaded or file is empty.")
        return map_widget

    # Data visualization outputs
    @render_widget
    def box_season_aet():
        df = process_data()
        df_grouped = df.groupby(["Season", "year"], as_index=False)["AET"].sum()
        fig = go.Figure()
        for year in df_grouped["year"].unique():
            trend_line = df_grouped[df_grouped["year"] == year].groupby("Season", as_index=False)["AET"].mean()
            fig.add_trace(go.Scatter(x=trend_line["Season"], y=trend_line["AET"], 
                                     mode='lines+markers', name=str(year)))
        return fig

    @render_widget
    def box_season_precip():
        df = process_data()
        df_grouped = df.groupby(["Season", "year"], as_index=False)["Rainfall"].sum()
        fig = go.Figure()
        for year in df_grouped["year"].unique():
            trend_line = df_grouped[df_grouped["year"] == year].groupby("Season", as_index=False)["Rainfall"].sum()
            fig.add_trace(go.Scatter(x=trend_line["Season"], y=trend_line["Rainfall"], 
                                     mode='lines+markers', name=str(year)))
        return fig

    @render_widget
    def box_season_ws():
        df = process_data()
        df_grouped = df.groupby(["Season", "year"], as_index=False)["Total_Water_Cubic_Meters"].sum()
        fig = go.Figure()
        for year in df_grouped["year"].unique():
            trend_line = df_grouped[df_grouped["year"] == year].groupby("Season", as_index=False)["Total_Water_Cubic_Meters"].mean()
            fig.add_trace(go.Scatter(x=trend_line["Season"], y=trend_line["Total_Water_Cubic_Meters"], 
                                     mode='lines+markers', name=str(year)))
        return fig

    @render_widget
    def yearly_precip():
        data1 = process_data()
        start_month = input.start_month()
        period = input.period()
        val = calculate_period_average(data1, start_month, period, 'Rainfall')
        df = pd.DataFrame({"Year": list(val.keys()), "Value": list(val.values())})
        fig = px.bar(df, x="Year", y="Value")
        fig.update_layout(xaxis_title="Year", yaxis_title="Precipitation")
        return fig

    @render_widget
    def yearly_evap():
        data1 = process_data()
        start_month = input.start_month()
        period = input.period()
        val = calculate_period_average(data1, start_month, period, 'AET')
        df = pd.DataFrame({"Year": list(val.keys()), "Value": list(val.values())})
        fig = px.bar(df, x="Year", y="Value")
        fig.update_layout(xaxis_title="Year", yaxis_title="Evapotranspiration")
        return fig

    @render_widget
    def yearly_ws():
        data1 = process_data()
        start_month = input.start_month()
        period = input.period()
        val = calculate_period_average(data1, start_month, period, 'Total_Water_Cubic_Meters')
        df = pd.DataFrame({"Year": list(val.keys()), "Value": list(val.values())})
        fig = px.bar(df, x="Year", y="Value")
        fig.update_layout(xaxis_title="Year", yaxis_title="Water Use (m3)")
        return fig

    @render_widget
    def bar_month_year_precip():
        df = process_data()
        if df.empty:
            return None
        # Agréger par mois et année
        monthly_data = df.groupby(['year', 'month'])['Rainfall'].sum().reset_index()
        fig = px.bar(monthly_data, x="month", y="Rainfall", color="year",
                    title="Monthly Precipitation by Year",
                    labels={"month": "Month", "Rainfall": "Precipitation (mm)", "year": "Year"})
        return fig

    @render_widget
    def bar_month_year_aet():
        df = process_data()
        if df.empty:
            return None
        # Agréger par mois et année
        monthly_data = df.groupby(['year', 'month'])['AET'].sum().reset_index()
        fig = px.bar(monthly_data, x="month", y="AET", color="year",
                    title="Monthly Evapotranspiration by Year",
                    labels={"month": "Month", "AET": "Evapotranspiration (mm)", "year": "Year"})
        return fig

    @render_widget
    def bar_month_year_ws():
        df = process_data()
        if df.empty:
            return None
        # Agréger par mois et année
        monthly_data = df.groupby(['year', 'month'])['Total_Water_Cubic_Meters'].sum().reset_index()
        fig = px.bar(monthly_data, x="month", y="Total_Water_Cubic_Meters", color="year",
                    title="Monthly Water Use by Year",
                    labels={"month": "Month", "Total_Water_Cubic_Meters": "Water Use (m³)", "year": "Year"})
        return fig

    @render_widget
    def results_plot():
        data = process_data()
        if data.empty:
            return None
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data.index, y=data['AET'], mode='lines', name='Mean AET', line=dict(color='green')
        ))
        fig.add_trace(go.Scatter(
            x=data.index, y=data['Rainfall'], mode='lines', name='Mean Precipitation', line=dict(color='blue')
        ))
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Mean Value",
            legend_title="Variables"
        )
        return fig

    @render_widget
    def results_plot_ws():
        data = process_data()
        if data.empty:
            return None
            
        # Agréger les données par année et saison
        seasonal_data = data.groupby(['year', 'Season'])['Total_Water_Cubic_Meters'].sum().reset_index()
        
        # Créer le graphique à barres empilées
        fig = px.bar(seasonal_data, 
                    x="year", 
                    y="Total_Water_Cubic_Meters", 
                    color="Season",
                    barmode="stack",
                    title="Annual Water Use by Season",
                    category_orders={"Season": ["Winter", "Spring", "Summer", "Fall"]})
        
        fig.update_layout(
            xaxis_title="Year",
            yaxis_title="Total Water Use (m³)",
            legend_title="Season",
            showlegend=True
        )
        return fig

    @output
    @render.table
    def merged_data_table():
        merged_df = process_data()

        # Réinitialiser l'index pour le transformer en colonne
        merged_df = merged_df.reset_index()

        # Renommer l'index si nécessaire
        merged_df.rename(columns={'index': 'Date'}, inplace=True)
        # merged_df["Date"] = pd.to_datetime(merged_df["Date"], errors='coerce')

        cols = [
            "Date",  # Utilisez 'Date' à la place de 'data.index'
            "AET",
            "Rainfall",
            "Total_Water_Cubic_Meters",            
        ]

        # Vérifier si les colonnes existent
        if merged_df is None or any(col not in merged_df.columns for col in cols):
            return None

        # Filtrer les colonnes et renvoyer le DataFrame
        merged_df = merged_df[cols]
        return merged_df.reset_index()

    @render.text
    def area_irrigated():
        data = process_data()
        if data.empty:
            return None
        var = data["Irrigated_Area_Hec"].mean().__round__(2)
        return f"{var} Hec"

    # Download data handlers
    @output
    @render.text
    def wapor_status():
        return status_text_wapor()

    status_text_wapor = reactive.Value("")

    @reactive.Effect
    @reactive.event(input.wapor_download)
    def _():
        start_year = int(input.wapor_start_year())
        end_year = int(input.wapor_end_year())
        start_month = int(input.wapor_start_month())
        end_month = int(input.wapor_end_month())

        if start_year > end_year or (start_year == end_year and start_month > end_month):
            status_text_wapor.set("Error: Start date must be before end date.")
        else:
            status_text_wapor.set("Downloading...")
            download_data(start_year, end_year, start_month, end_month)
            status_text_wapor.set("Download completed!")

    @output
    @render.text
    def chirps_status():
        return status_text_chirps()

    status_text_chirps = reactive.Value("")

    @reactive.Effect
    @reactive.event(input.chirps_download)
    def _():
        start_year = int(input.chirps_start_year())
        end_year = int(input.chirps_end_year())
        start_month = int(input.chirps_start_month())
        end_month = int(input.chirps_end_month())

        if start_year > end_year or (start_year == end_year and start_month > end_month):
            status_text_chirps.set("Error: Start date must be before end date.")
        else:
            status_text_chirps.set("Downloading and extracting...")
            download_and_extract_data(start_year, end_year, start_month, end_month)
            status_text_chirps.set("Download and extraction completed!")

    # Processing handlers
    @reactive.Effect
    @reactive.event(input.start_process_aet_01)
    async def process_aet():
        year = input.year_aet_01()
        month = input.month_aet_01()
        dekad = input.dekad_aet_01()
        nmfic = f"{year}-{month}-{dekad}"
        try:
            status_messager_aet_01.set("Traitement en cours...")
            await asyncio.get_event_loop().run_in_executor(None, process_nmfic_aet_01, nmfic)
            status_messager_aet_01.set("Traitement terminé avec succès!")
        except Exception as e:
            status_messager_aet_01.set(f"Erreur: {str(e)}")

    @reactive.Effect
    @reactive.event(input.start_process_01_chirps)
    async def process_chirps():
        year = int(input.year_01_chirps())
        month = int(input.month_01_chirps())
        dekad = int(input.dekad_01_chirps()[1])
        try:
            status_messager_01_chirps.set("Traitement en cours...")
            await asyncio.get_event_loop().run_in_executor(None, process_nmfic_01_chirps, year, month, dekad)
            status_messager_01_chirps.set("Traitement terminé avec succès!")
        except Exception as e:
            status_messager_01_chirps.set(f"Erreur: {str(e)}")

    @reactive.Effect
    @reactive.event(input.run_diff)
    def process_difference():
        year = input.year_diff()
        month = input.month_diff()
        dekad = input.dekad_diff()
        try:
            status_messager_diff.set("Traitement en cours...")
            et_file = os.path.join("outputs/AET_dekad_brut_m/", get_et_filename(year, month, dekad))
            rainfall_file = os.path.join("outputs/Chirps_dekad_brut_m/", get_rainfall_filename(year, month, dekad))
            output_file = os.path.join("outputs/difirence_dekad_brut_m/", f"{year}-{str(month).zfill(2)}-D{dekad}.tif")
            calculate_and_write_difference(et_file, rainfall_file, output_file, gdf)
            status_messager_diff.set("Traitement terminé avec succès!")
        except Exception as e:
            status_messager_diff.set(f"Erreur: {str(e)}")

        

################################# End Diff ###############################################################################

############### Missing Data ##############


    @output
    @render.table
    def directory_analysis():
        directories = {
            'AET': aet_dir,
            'CHIRPS': chirps_dir,
            'Différence': diff_dir
        }

        results = []
        for name, directory in directories.items():
            start_date, end_date, missing_dates = analyze_directory_dates(directory)
            missing_count = len(missing_dates) if missing_dates else 0
            results.append({
                'Type': name,
                'Date de début': start_date or 'N/A',
                'Date de fin': end_date or 'N/A',
                'Nombre de dekads manquantes': missing_count,
                'Dekads manquantes': ', '.join(missing_dates) if missing_dates else 'Aucune'
            })

        return pd.DataFrame(results)

    @session.download(filename="data.csv")
    def download_csv():
        # Convert DataFrame to CSV
        data = process_data()
        csv_data = data.to_csv(index=False)
        yield csv_data

#######################################
# Création de l'application Shiny
app = App(app_ui, server, static_assets=pathlib.Path(__file__).parent/"www")

# Exécution de l'application
# if __name__ == "__main__":
#    app.run()
