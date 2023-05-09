# Exploration of a Methodology for Hyperspectral Band Selection using XGBoost and PCA

## Idea of this project
The classification of hyperspectral images involves utilizing multiple features to identify and classify each pixel's content accurately. PCA is a mathematical technique that reduces the dimensionality of hyperspectral images by identifying the most significant patterns of variability.
    
This paper aims to evaluate the effectiveness of Principal Component Analysis (PCA) based band selection for hyperspectral image classification using XGBoost feature importance scores.
    
To ensure that PCA uses the best set of bands to explain the variation, bands are proactively removed in the feature selection chain before dimensionality reduction. 
    
The paper details the techniques employed, including XGBoost feature importance scoring. Additionally, a flowchart is included to illustrate the process in its entirety.
    
The results demonstrate that the algorithm effectively reduces the number of bands while preserving crucial spectral data. This results in a slight decrease in the number of principal components, indicating the algorithm's effectiveness in locating and removing unwanted bands. The performance comparison across all four datasets shows a slight gain in accuracy, with some datasets having a more pronounced improvement. The suggested methodology generates a list of bands to be removed before PCA to improve classification accuracy.
    
Overall, this method shows promise in improving the accuracy of HS image classification and offers opportunities for future research, such as investigating alternative feature importance analysis techniques and extending the approach to other tree-based machine learning algorithms.

## Objectives

  1 -To investigate the effectiveness of incorporating preemptive band removal into Principal Component Analysis (PCA), with the main focus on optimizing accuracy over the reliability of explained variance.

  2 -To employ XGBoost to generate scores for each band and to utilize these scores to determine the optimal sequence for band removal before PCA.

  3 -To comprehensively compare the proposed algorithm with a conventional PCA application, providing a detailed, step-by-step description of the feature selection process for potential future research and analysis.

## Project Structure

```
HS_Image
  data
    logs
    processed_images
    raw
  notebooks
  utils
```

## About Datasets

**A. Salinas Dataset:** Using the 224-band AVIRIS sensor, the Salinas dataset was collected over the Salinas Valley in California. It has a high spatial resolution of 3.7-meter pixels. Vegetables, bare soils, and grape fields are depicted in the 512 lines by 217 samples. The collection has 224 bands and 16 categories for ground objects.

[Salinas (26.3 MB)](https://www.ehu.eus/ccwintco/uploads/f/f1/Salinas.mat) | [Salinas groundtruth (4.2 KB)](https://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat)

**B. Indian Pines Dataset:** This AVIRIS dataset was captured over the Indian Pines region in the northwestern part of Indiana, USA, and is available from the same website as the Salinas dataset. The original image size was 145 x 145 pixels with a spatial resolution of 20m and comprises 220 bands and 16 different ground object categories.

[Indian Pines (6.0 MB)](https://www.ehu.eus/ccwintco/uploads/2/22/Indian_pines.mat) | [Indian Pines groundtruth (1.1 KB)](https://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat)

**C. Pavia Centre Dataset:** Captured by a sensor known as the Reflective Optics System Imaging Spectrometer (ROSIS-3) over the city of Pavia, northern Italy. It is also available on the same website as the Salinas dataset. The image size is 1096 × 1096 pixels with a spatial resolution of 1.3 m, 102 bands in the range of 0.43–0.86 μm. Ground truth comprises nine classes.

[Pavia Centre (123.6 MB)](https://www.ehu.eus/ccwintco/uploads/e/e3/Pavia.mat) | [Pavia Centre groundtruth (34.1 KB)](https://www.ehu.eus/ccwintco/uploads/5/53/Pavia_gt.mat)

**D. Pavia University Dataset:** Captured by the same sensor of Pavia Centre. It is also available on the same website as the Salinas dataset. The image size is 610 × 340 pixels with a spatial resolution of 1.3 m, 103 bands in the range of 0.43–0.86 μm. Ground truth comprises nine classes.

[Pavia University (33.2 MB)](https://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat) | [Pavia University groundtruth (10.7 KB)](https://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat)

These datasets are from the official website of the  [official website of the Computational Intelligence Group of the University of the Basque Country (UPV/EHU)](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes).

## Results

- The total number of bands for each dataset after removal is compared with the original number of bands in the following table.
<p align="left">
    <img src="https://github.com/RealXun/Methodology_for_Hyperspectral_Band_Selection/blob/main/data/processed_images/band_reduction_results.png" width="500">
</p>

|                  | Nº of bands before removal | Nº of bands after removal |
|------------------|:--------------------------:|:-------------------------:|
| Salinas          | 224                        | 182                       |
| Indian Pines     | 220                        | 200                       |
| Pavia Centre     | 102                        | 81                        |
| Pavia University | 103                        | 62                        |




- Comparison between the number of components obtained without the algorithm and the number of components with the proposed methodology
<p align="left">
    <img src="https://github.com/RealXun/Methodology_for_Hyperspectral_Band_Selection/blob/main/data/processed_images/pca_results.png" width="500">
</p>

|                  | Nº principal components | Nº principal components with proposed methodology |
|------------------|:-----------------------:|:-------------------------------------------------:|
| Salinas          | 6                       | 6                                                 |
| Indian Pines     | 69                      | 64                                                |
| Pavia Centre     | 14                      | 9                                                 |
| Pavia University | 16                      | 11                                                |




- Accuracy obtained without algorithm compared to with algorithm.
<p align="left">
    <img src="https://github.com/RealXun/Methodology_for_Hyperspectral_Band_Selection/blob/main/data/processed_images/accuracy_results.png" width="500">
</p>


|                  | Accuracy without algorithm | Accuracy with algorithm |
|------------------|:--------------------------:|:-----------------------:|
| Salinas          | 0.889543                   | 0.895174                |
| Indian Pines     | 0.720710                   | 0.729007                |
| Pavia Centre     | 0.892378                   | 0.898099                |
| Pavia University | 0.837373                   | 0.856461                |


- The index of each removed band for all four datasets.
<p align="left">
    <img src="https://github.com/RealXun/Methodology_for_Hyperspectral_Band_Selection/blob/main/data/processed_images/deleted_bands.png" width="500">
</p>

|                  |                                                                                                 Removed bands                                                                                                |
|------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| Salinas          | [147, 146, 221, 157, 102, 220, 36, 170, 38, 5, 134, 185, 154,  145, 107, 0, 125, 222, 211, 223, 204, 44, 175, 68, 162, 156,  177, 133, 158, 111, 63, 174, 219, 215, 150, 106, 164, 193, 109,  148, 206, 149] |
| Indian Pines     |                                                        [192, 159, 12, 180, 103, 108, 122, 75, 165, 127, 109, 71, 124,  89, 30, 182, 67, 63, 173, 163]                                                        |
| Pavia Centre     |                                                                 [1, 0, 2, 3, 8, 6, 4, 89, 92, 5, 38, 57, 42, 87, 81, 78, 9, 80, 75,  88, 26]                                                                 |
| Pavia University |                        [1, 0, 2, 3, 8, 4, 29, 52, 9, 49, 5, 57, 6, 43, 38, 73, 14, 26, 11,  102, 55, 22, 93, 94, 45, 98, 100, 80, 89, 46, 19, 18, 79, 51,  56, 50, 37, 59, 95, 36, 16]                       |

## Conclusion and Future works Section Writing

This study proposes a band selection approach for HS image classification that combines XBoost feature importance analysis with PCA to improve accuracy while reducing the computational cost. By removing the least important bands and applying PCA to the remaining ones, the proposed methodology is able to retain critical information while also reducing the dimensionality of the dataset. This approach can have a significant impact in various fields where HS imaging is commonly used, such as agriculture and medical imaging. It essentially helps to improve classification during the previous feature selection and dimensionality reduction phases, providing a flat increase in accuracy before hyperparameter tuning and model training, which in turn can lead to improved decision-making and overall better results.

The use of XGBoost with the histogram algorithm is an important step in the methodology, as it reduces the computational cost by approximating the gradients using histograms. This is significant because the process trains multiple models, so any reduction in terms of computational cost when training has a large effect.
Throughout this research work, numerous challenges have been addressed. During experimentation, it has been necessary to determine an appropriate amount of variance to retain in the PCA transformation. Since HS images often contain a large number of highly correlated spectral bands, it is important to retain enough variance to capture the important information. This has been solved by observing different variance thresholds and selecting the one that produces a high accuracy while still requiring a manageable number of components. Finding the absolute optimal variance threshold is not critical for the experimentation since the aim is to use the algorithm to improve accuracy relative to applying PCA and XGBoost in a conventional way with the same parameters. Another challenge has been reproducibility and coherence through iterations since multiple XGBoost models and splits are used during the process. This has been solved by using a fixed seed that remains constant across the algorithm loops.

The proposed methodology offers several future opportunities for research. One possibility is to explore alternative feature importance analysis techniques to identify the most important bands before the removal process. Another avenue for research is to investigate the impact of different variance thresholds. The list of removed bands is also worthy of further research since looking at the reasons why removing them increases accuracy can help understand the underlying difficulties that those bands cause to PCA during dimensionality reduction, which ultimately leads to a reduction in accuracy. Additionally, this approach could be extended to other machine learning algorithms that are tree-based, namely decision trees or random forests.


## Software implementation

All source code used to generate the results and figures in the paper are in the `notebooks` folder.

The data used in this study is provided in "raw" folder (a folder inside "data" folder).

Results generated by the code are saved in "logs" folder (a folder inside "data" folder).

The images generated using "" file are saved inside "processed_images" (a folder inside "data" folder).

## Dependencies and libraries imported

You'll need a working Python environment to run the code.

The required dependencies are specified in the file `requirenments.txt`.

```
NumPy: A powerful Python library for numerical computing.
SciPy: A collection of mathematical algorithms and functions built on NumPy.
XGBClassifier: A gradient boosting machine learning algorithm provided by the XGBoost library.
Pandas: A versatile library for data manipulation and analysis.
Scikit-learn (sklearn): A comprehensive library of machine learning algorithms and tools for Python.
```

## Reproducing the results

Clone this repository to have the exact same structure or use this link https://github.com/RealXun/Methodology_for_Hyperspectral_Band_Selection.git

$${\color{red}Make  \space  sure \space  all \space  datasets \space  mentioned \space  before \space  are \space  downloaded \space  and \space  saved \space  in \space  "raw" \space  folder \space  (a \space  folder \space  inside \space  "data" \space  folder).}$$
:warning: Make sure all datasets mentioned before are downloaded and saved in "raw" folder (a folder inside "data" folder) :warning:

There are two easy ways to run the code and see the results.

1- Using VisualStudioCode and run a Jupyter notebook (Recommended): The Jupyter notebook is called "PCA_band_removal.ipynb" and is located inside the folder names "notebooks": The notebook is divided into cells (some have text while other have code). Each cell can be executed using `Shift + Enter`.
Executing text cells does nothing and executing code cells runs the code and produces it's output. To execute the whole notebook, run all cells in order.

2- Running the python file named "hsi_pca.py": It will show you a short menu where you can choose which dataset you want to use.
