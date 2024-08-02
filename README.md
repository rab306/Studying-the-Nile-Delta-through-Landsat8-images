# Nile Delta Landsat 8 Analysis

This project analyzes the erosion of the Mediterranean Sea coast in the Nile Delta region and assesses changes in agricultural lands using satellite images from the Landsat 8 mission. The analysis focuses on images with a resolution of 30m/pixel and utilizes K-means clustering for classification.

## Project Overview

The script performs the following tasks:
1. **Load Landsat Images**: Load images from different years (2014, 2017, 2024) and their respective classification maps.
2. **Preprocess Data**: Exclude zero-data pixels and ensure consistency in image dimensions.
3. **Calculate Remote Sensing Indices**: Compute indices such as NDVI, NDWI, AWEI, NDBI, UI, and BUI to analyze vegetation, water bodies, and built-up areas.
4. **Plot and Save Results**:
   - Generate subplots of calculated indices with coordinates.
   - Create histograms for each index to visualize distribution.
   - Calculate and visualize areas of different land cover classes over time.
   - Perform and visualize change detection between different years.

## Installation

Ensure you have the following Python libraries installed:

- `rasterio`
- `numpy`
- `matplotlib`
- `pandas`

You can install them using pip:

```bash
pip install rasterio numpy matplotlib pandas
```

## Usage

1. **Load Landsat Imagery**: Update the `load_image` function to include paths to your Landsat images and classification maps.

2. **Preprocess Images**: The script handles zero-data pixels and aligns images from different years.

3. **Calculate Indices**: Modify or add indices as needed for your analysis.

4. **Plot Results**: Customize plotting functions as necessary to suit your needs.

Run the script:

```bash
python your_script_name.py
```

The script will generate several output files including:

- `ndvi_subplot.png`
- `ndwi_subplot.png`
- `modified_ndwi_subplot.png`
- `awei_subplot.png`
- `awei_shadow_subplot.png`
- `ndbi_subplot.png`
- `ui_subplot.png`
- `bu_subplot.png`
- `ndvi_histograms.png`
- `ndwi_histograms.png`
- `modified_ndwi_histograms.png`
- `awei_histograms.png`
- `awei_shadow_histograms.png`
- `ndbi_histograms.png`
- `ui_histograms.png`
- `bui_histograms.png`
- `area_claculation_barPlot.png`
- `change_detection_barPlot.png`

## Notes

- Ensure your Landsat images and classification maps are in the correct format and resolution.
- Modify file paths in the script according to your local directory structure.
- The script assumes a consistent image size across all years; adjust as needed for your data.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

