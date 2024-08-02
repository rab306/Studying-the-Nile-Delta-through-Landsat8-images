
"""This script is used to analyze the erosion of mediterranean sea coast in the nile delta region
     along with assesing the changes in the agricultural lands in the delta using satellite
     images of landsat8 mission with resolution of 30m/pixel, the classification maps of these images resulted from
     K-means clustering"""


# Importing libraries
import rasterio   # rasterio is necessary for Preservation the spatial information in the raster data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd


# Loading Landsat imagery
def load_image(file_path):
    with rasterio.open(file_path) as src:  # src is rasterio object that its index starts from 1
        return src.read(), src.profile  # src.read() returns a numpy array of the image, src.profile returns the metadata of the image

# load the original images and classification maps
image_14, profile_14 = load_image('results\2014.img')
image_17, profile_17 = load_image('results\2019.img')
image_24, profile_24 = load_image('results\2024.img')
classification_2014, profile_2014 = load_image('results\class2014.img')
classification_2017, profile_2017 = load_image('results\class2019.img')
classification_2024, profile_2024 = load_image('results\class2024.img')



# Working with original images
# Excluding pixels with zero data
image_14 = np.where(image_14 != 0, image_14, 1e-9)
image_17 = np.where(image_17 != 0, image_17, 1e-9)
image_24 = np.where(image_24 != 0, image_24, 1e-9)


# Ensure images are of the same size
min_rows = min(image_14.shape[1], image_17.shape[1], image_24.shape[1])
min_cols = min(image_14.shape[2], image_17.shape[2], image_24.shape[2])

image_14 = image_14[:, :min_rows, :min_cols]
image_17 = image_17[:, :min_rows, :min_cols]
image_24 = image_24[:, :min_rows, :min_cols]
print(image_14.shape, image_17.shape, image_24.shape)

# Extracting blue, green, red, NIR, SWIR-1 and SWIR-2
blue_14 = image_14[1]   # Band 2
green_14 = image_14[2]  # Band 3
red_14 = image_14[3]    # Band 4
nir_14 = image_14[4]    # Band 5
swir_14 = image_14[5]   # Band 6
swir2_14 = image_14[6]   # Band 7

blue_17 = image_17[1]
green_17 = image_17[2]
red_17 = image_17[3]
nir_17 = image_17[4]
swir_17 = image_17[5]
swir2_17 = image_17[6]


blue_24 = image_24[1]
green_24 = image_24[2]
red_24 = image_24[3]
nir_24 = image_24[4]
swir_24 = image_24[5]
swir2_24 = image_24[6]


# Calculation of different remote sensing indices
def calculate_index(band_1, band_2):
    index = (band_1 - band_2) / (band_1 + band_2)
    return index


# Calculating NDVI
ndvi_14 = calculate_index(nir_14, red_14)
ndvi_17 = calculate_index(nir_17, red_17)
ndvi_24 = calculate_index(nir_24, red_24)


# Calculating NDWI
ndwi_14 = calculate_index(green_14, nir_14)
ndwi_17 = calculate_index(green_17, nir_17)
ndwi_24 = calculate_index(green_24, nir_24)


# Calculating modified NDWI in 2 ways
ndwi_mod_14 = calculate_index(green_14, swir_14)
ndwi_mod_17 = calculate_index(green_17, swir_17)
ndwi_mod_24 = calculate_index(green_24, swir_24)

ndwi_mod2_14 = calculate_index(green_14, swir2_14)
ndwi_mod2_17 = calculate_index(green_17, swir2_17)
ndwi_mod2_24 = calculate_index(green_24, swir2_24)


# Calculating Automated Water Extraction Index (AWEI) with no shadows
awei_14 = 4 * (green_14 - swir_14) - (0.25 * nir_14 + 2.75 * swir_14)
awei_17 = 4 * (green_17 - swir_17) - (0.25 * nir_17 + 2.75 * swir_17)
awei_24 = 4 * (green_24 - swir_24) - (0.25 * nir_24 + 2.75 * swir_24)


# Calculate Automated Water Extraction Index (AWEI) with shadows
awei_shadow_14 = blue_14 + 2.5 * green_14 - 1.5 * (nir_14 + swir_14) - 0.25 * swir2_14
awei_shadow_17 = blue_17 + 2.5 * green_17 - 1.5 * (nir_17 + swir_17) - 0.25 * swir2_17
awei_shadow_24 = blue_24 + 2.5 * green_24 - 1.5 * (nir_24 + swir_24) - 0.25 * swir2_24


# Calculating NDBI
ndbi_14 = calculate_index(swir_14, nir_14)
ndbi_17 = calculate_index(swir_17, nir_17)
ndbi_24 = calculate_index(swir_24, nir_24)


# Calculating urban index (UI)
ui_14 = calculate_index(swir2_14, nir_14)
ui_17 = calculate_index(swir2_17, nir_17)
ui_24 = calculate_index(swir2_24, nir_24)


# Calculating Built-up Index (BUI)
bui_14 = ndbi_14 - ndvi_14
bui_17 = ndbi_17 - ndvi_17
bui_24 = ndbi_24 - ndvi_24



# Plotting the indices
# The plot will use the real coordinates of the images stored in the metadata
# These coordinates are transformed using affine transformation matrix
def plot_index_subplots_with_coords(data_list, years, index_name, cmap, transform, save_path):
    fig, axs = plt.subplots(len(data_list), 1, figsize=(10, 6 * len(data_list)))

    for i, (data, year) in enumerate(zip(data_list, years)):
        ax = axs[i]
        rows, cols = data.shape
        x = transform[2] + np.arange(cols) * transform[0]
        y = transform[5] + np.arange(rows) * transform[4]
        im = ax.imshow(data, cmap=cmap, extent=[x.min(), x.max(), y.min(), y.max()])

        # Format y-axis ticks to display whole numbers
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: '{:.0f}'.format(val)))

        ax.set_title(f'{index_name} {year}')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.grid(True, linestyle='--', linewidth=0.5, color='gray')

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(index_name)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()



plot_index_subplots_with_coords(
    data_list=[ndvi_14, ndvi_17, ndvi_24],
    years=[2014, 2017, 2024],
    index_name='NDVI',
    cmap='RdYlGn',
    transform=profile_14['transform'],
    save_path=r'ndvi_subplot.png'
)

plot_index_subplots_with_coords(
    data_list=[ndwi_14, ndwi_17, ndwi_24],
    years=[2014, 2017, 2024],
    index_name='NDWI',
    cmap='RdBu',
    transform=profile_14['transform'],
    save_path=r'ndwi_subplot.png'
)

plot_index_subplots_with_coords(
    data_list=[ndwi_mod_14, ndwi_mod_17, ndwi_mod_24],
    years=[2014, 2017, 2024],
    index_name='First NDWI',
    cmap='RdBu',
    transform=profile_14['transform'],
    save_path=r'modified_ndwi_subplot.png'
)

plot_index_subplots_with_coords(
    data_list=[ndwi_mod2_14, ndwi_mod2_17, ndwi_mod_24],
    years=[2014, 2017, 2024],
    index_name='Second NDWI',
    cmap='RdBu',
    transform=profile_14['transform'],
    save_path=r'_modified2_ndwi_subplot.png'
)

plot_index_subplots_with_coords(
    data_list=[awei_14, awei_17, awei_24],
    years=[2014, 2017, 2024],
    index_name='AWEI',
    cmap='RdBu',
    transform=profile_14['transform'],
    save_path=r'awei_subplot.png'
)

plot_index_subplots_with_coords(
    data_list=[awei_shadow_14, awei_shadow_17, awei_shadow_24],
    years=[2014, 2017, 2024],
    index_name='AWEI Shadow',
    cmap='RdBu',
    transform=profile_14['transform'],
    save_path=r'awei_shadow_subplot.png'
)


plot_index_subplots_with_coords(
    data_list=[ndbi_14, ndbi_17, ndbi_24],
    years=[2014, 2017, 2024],
    index_name='NDBI',
    cmap='bwr',
    transform=profile_14['transform'],
    save_path=r'ndbi_subplot.png'
)


plot_index_subplots_with_coords(
    data_list=[ui_14, ui_17, ui_24],
    years=[2014, 2017, 2024],
    index_name='UI',
    cmap='bwr',
    transform=profile_14['transform'],
    save_path=r'ui_subplot.png'
)

plot_index_subplots_with_coords(
    data_list=[bui_14, bui_17, bui_24],
    years=[2014, 2017, 2024],
    index_name='BUI',
    cmap='seismic',
    transform=profile_14['transform'],
    save_path=r'bu_subplot.png'
)



# plotting histograms for indices
def plot_histograms(indices_data, metric, years, color, save_path):
    plt.figure(figsize=(8, 12))

    for i, (indices, year) in enumerate(zip(indices_data, years)):
        plt.subplot(3, 1, i + 1)
        plt.hist(indices.flatten(), bins=50, alpha=1, color=color, label=f'{metric} {year}')
        plt.xlabel(f'{metric} Value')
        plt.ylabel('Frequency')
        plt.title(f'{metric} Histograms for {year}')
        plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


plot_histograms(
    indices_data=[ndvi_14, ndvi_17, ndvi_24],
    metric='NDVI',
    years=[2014, 2017, 2024],
    color='green',
    save_path=r'ndvi_histograms.png'
)


plot_histograms(
    indices_data=[ndwi_14, ndwi_17, ndwi_24],
    metric='NDWI',
    years=[2014, 2017, 2024],
    color='blue',
    save_path=r'ndwi_histograms.png'
)


plot_histograms(
    indices_data=[ndwi_mod_14, ndwi_mod_17, ndwi_mod_24],
    metric='MNDWI',
    years=[2014, 2017, 2024],
    color='blue',
    save_path=r'modified_ndwi_histograms.png'
)


plot_histograms(
    indices_data=[ndwi_mod2_14, ndwi_mod2_17, ndwi_mod2_24],
    metric='MNDWI-2',
    years=[2014, 2017, 2024],
    color='blue',
    save_path=r'modified2_ndwi_histograms.png'
)


plot_histograms(
    indices_data=[awei_14, awei_17, awei_24],
    metric='AWEI',
    years=[2014, 2017, 2024],
    color='blue',
    save_path=r'awei_histograms.png'
)


plot_histograms(
    indices_data=[awei_shadow_14, awei_shadow_17, awei_shadow_24],
    metric='AWEI Shadow',
    years=[2014, 2017, 2024],
    color='blue',
    save_path=r'awei_shadow_histograms.png'
)


plot_histograms(
    indices_data=[ndbi_14, ndbi_17, ndbi_24],
    metric='NDBI',
    years=[2014, 2017, 2024],
    color='red',
    save_path=r'ndbi_histograms.png'
)


plot_histograms(
    indices_data=[ui_14, ui_17, ui_24],
    metric='UI',
    years=[2014, 2017, 2024],
    color='red',
    save_path=r'ui_histograms.png'
)

plot_histograms(
    indices_data=[bui_14, bui_17, bui_24],
    metric='BUI',
    years=[2014, 2017, 2024],
    color='red',
    save_path=r'bui_histograms.png'
)

# Working with classification maps
# Ensure classification images are of the same size
min_rows = min(classification_2014.shape[0], classification_2017.shape[0], classification_2024.shape[0])
min_cols = min(classification_2014.shape[1], classification_2017.shape[1], classification_2024.shape[1])

classification_2014 = classification_2014[:min_rows, :min_cols]
classification_2017 = classification_2017[:min_rows, :min_cols]
classification_2024 = classification_2024[:min_rows, :min_cols]

# Exclude unclassified pixels (class index zero)
classification_2014 = np.where(classification_2014 != 0, classification_2014, np.nan)
classification_2017 = np.where(classification_2017 != 0, classification_2017, np.nan)
classification_2024 = np.where(classification_2024 != 0, classification_2024, np.nan)

# Calculating Areas
def calculate_area(classification, pixel_size):
    unique, counts = np.unique(classification, return_counts=True)
    areas = dict(zip(unique, counts * pixel_size**2 / 1e6))  # km²
    return areas

pixel_size = 30    # meters
areas_2014 = calculate_area(classification_2014, pixel_size)
areas_2017 = calculate_area(classification_2017, pixel_size)
areas_2024 = calculate_area(classification_2024, pixel_size)

# Combining areas into a single DataFrame
areas_df = pd.DataFrame([areas_2014, areas_2017, areas_2024], index=[2014, 2017, 2024])

# Excluding unclassified pixels (class index zero)
areas_df = areas_df.drop(columns=[0], errors='ignore')
areas_df = areas_df.iloc[:, :-3]

areas_df.columns = ['Water Bodies', 'Land Areas', 'Cultivated Areas']
print(areas_df)

# Change Detection Analysis
change_detection = pd.DataFrame()
change_detection['2014-2017'] = areas_df.loc[2017] - areas_df.loc[2014]
change_detection['2017-2024'] = areas_df.loc[2024] - areas_df.loc[2017]
change_detection['2014-2024'] = areas_df.loc[2024] - areas_df.loc[2014]

# Filter out NaN values
change_detection = change_detection.dropna(how='all', axis=1).dropna(how='all')
print(change_detection)

# Visualization as grouped bar plot
plt.figure(figsize=(12, 8))
bar_width = 0.2
years = areas_df.index
positions = np.arange(len(years))

# Plot bars for each class
colors = ['blue', 'red', 'green']
for i, (class_label, color) in enumerate(zip(areas_df.columns, colors)):
    plt.bar(positions + i * bar_width, areas_df[class_label], width=bar_width, label=class_label, color=color)

plt.xlabel('Year')
plt.ylabel('Area (km²)')
plt.title('Area Calculation Over Time')
plt.xticks(positions + bar_width, years)  # Center the ticks
plt.legend()
plt.savefig(r'area_claculation_barPlot.png', dpi=300, bbox_inches='tight')
plt.show()

# Bar plot for change detection
colors = ['purple', 'orange', 'cyan']
ax = change_detection.plot(kind='bar', figsize=(12, 6), color=colors)
ax.set_xlabel('Land Cover Class')
ax.set_ylabel('Change in Area (km²)')
ax.set_title('Change Detection Between Years')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='right')
plt.savefig(r'change_detection_barPlot.png', dpi=300, bbox_inches='tight')
plt.show()



