# OCR Project for Mathematical Expression Recognition

<!-- Short project description -->
A project focused on Optical Character Recognition (OCR) for mathematical expressions. This work leverages a nested JSON dataset (with fields like `latex`, `uuid`, `unicode_str`, etc.) and uses Spark for preprocessing before training recognition models.

## Table of Contents

- [Overview](#overview)
- [Dataset Description](#dataset-description)
- [Data Preprocessing](#data-preprocessing)
- [Merging Data](#merging-data)
- [Model Training](#model-training)
- [Installation](#installation)
- [Usage](#usage)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview

This project aims to develop an OCR system capable of recognizing and parsing handwritten mathematical expressions. The dataset includes both human-written and synthetic samples and provides detailed metadata such as LaTeX representations, unique identifiers, and nested image data (including bounding boxes and mask information).

Key features of the project:
- Extraction of nested JSON fields (e.g., `image_data.full_latex_chars`).
- Data transformation using PySpark.
- Merging newly derived columns back into the original DataFrame.
- Model training for OCR using both traditional and modern architectures.

## Dataset Description

The dataset is provided in JSON format and includes the following key fields:
- **latex**: The LaTeX expression of the mathematical formula.
- **uuid**: Unique identifier for each sample.
- **unicode_str** and **unicode_less_curlies**: Variants of the expression in Unicode.
- **image_data**: A nested JSON object that includes:
  - `full_latex_chars`
  - `visible_latex_chars`
  - `visible_char_map`
  - `width`, `height`, `depth`
  - Bounding box details: `xmins`, `xmaxs`, `ymins`, `ymaxs`
  - Raw bounding boxes: `xmins_raw`, `xmaxs_raw`, `ymins_raw`, `ymaxs_raw`
  - `png_masks`
- **font**: Font information.
- **filename**: Original filename of the sample.

Example Spark DataFrame preview:

| filename | font   | image_data | latex                    | unicode_less_curlies          | unicode_str                | uuid  | ... |
|----------|--------|------------|--------------------------|-------------------------------|----------------------------|-------|-----|
| sample1  | Oct_011| { ... }   | `\lim_{a\to\frac{...}`    | `ƀaƄಭೃ4ಭಭdda...`             | `ƀ{aƄಭ{ೃ}{4}}...`          | ...   | ... |

## Data Preprocessing

Data processing is done using PySpark. Key steps include:
- **Extracting Nested Arrays:**  
  Use `select` to access nested fields without exploding the data, for example:
  ```python
  from pyspark.sql.functions import col
  
  # Select the full array of LaTeX characters without exploding
  df = df.select("image_data.full_latex_chars")
  df.show(truncate=False)
# Dataset Splitting and Preparation Process

This repository uses a multi-stage process to ingest, split, and prepare the handwritten mathematical expression dataset for segmentation and model training. Below is a summary of the key steps in the pipeline:

## 1. Accessing the Data

- **Local and Cloud Access:**  
  - The code uses glob patterns to identify local JSON files (e.g., `"archive/batch_*/json/kaggle_data_*.json"`).
  - The dataset is also downloaded from Kaggle using the `kagglehub` package.

- **Batch Organization:**  
  - Data is organized into batches. For each batch, the script collects the JSON metadata files and the corresponding image files.

## 2. Creating the Spark DataFrame

- **Unioning JSON Files:**  
  - An empty Spark DataFrame is created using a schema derived from one sample JSON.
  - Each JSON file is read and unioned into a single DataFrame to consolidate all the metadata.

- **Casting Nested Columns:**  
  - The nested JSON fields (e.g., `full_latex_chars`, `visible_latex_chars`, bounding box coordinates, PNG masks) are cast into proper Spark types such as arrays and floats.
  - This ensures the data is structured correctly for further processing.

## 3. Cleaning and Reducing the DataFrame

- **Dropping Unnecessary Columns:**  
  - After unpacking and casting the necessary fields, the original nested column (`image_data`) and other unused fields (e.g., `unicode_less_curlies`, `unicode_str`) are dropped.  
  - This leaves a clean DataFrame (`image_props`) containing only the essential metadata for image segmentation.

## 4. Locating Image Files

- **Mapping Metadata to Images:**  
  - A helper function scans the image directories and matches the filename from the metadata with the actual image file stored on disk.
  - The result is a mapping between the Spark DataFrame rows and the corresponding image file paths.

## 5. Preparing Data for Segmentation

- **Segmentation Functions:**  
  - **Preprocessing:**  
    - Images are preprocessed (e.g., resized, thresholded) to generate binary images.
  - **Bounding Box Extraction:**  
    - Contours are detected from the binary images, and bounding boxes are computed for each symbol.
  - **Optional Mask Application:**  
    - PNG masks are decoded and overlaid on the extracted image regions to refine segmentation.
  - **Extraction & Saving:**  
    - Each segmented symbol is extracted and saved in output directories named after their corresponding LaTeX labels.

## 6. Overall Pipeline Flow

1. **Data Ingestion:**  
   - Read multiple JSON files and union them into one Spark DataFrame.
2. **Data Cleaning:**  
   - Cast nested fields into appropriate data types and drop unnecessary columns.
3. **Image Mapping:**  
   - Locate the corresponding background images using the filename field.
4. **Segmentation & Annotation:**  
   - Apply segmentation on each image using bounding box and mask data.
   - Write out segmented images into folders corresponding to their LaTeX labels.

This multi-stage process ensures that the raw, heterogeneous data is split and prepared in a scalable manner using Spark, making it ready for subsequent segmentation and model training steps.
