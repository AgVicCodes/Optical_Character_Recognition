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
