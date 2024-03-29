# Mapillary Points of Interest Neighborhood Street-level Imagery: A Potential Dataset for Neighborhood Analytics in Cities

### Dataset: MPOINSLI Mapillary POI-Neighborhood Street-Level Images.
MPOINSLI Datasets related to this article can be found at [here](https://doi.org/10.5281/zenodo.7618831), an open-source data repository hosted at Zenod available for academic research.


**Abstract**:

The Sustainable Development Goals of the United Nations promote sustainable urban development to make cities more economically and socially liveable. Points of Interest (POIs) such as commercial properties and healthcare facilities are significant markers for these goals. Street-view images are becoming increasingly important for capturing cities' streetscapes. Existing studies provide city-level images, while there are few studies that provide images in the vicinity of certain POIs. Therefore, this paper develops a framework for filtering images so that a portion of a given POI is visible in their field of view (FOV). We contribute with Mapillary POI-Neighborhood Street-Level Images (MPOINSLI) dataset, a large street-view image of POIs and their neighborhood in New York City. First, all the images within a 35-meter radius of certain POIs are filtered. Then, the intersection technique is utilized to determine if the cameras' FOV triangular polygons intersect the POIs' polygons. Using 11,126 POIs from SafeGraph's Geometry and Place datasets in conjunction with 875,592 Mapillary images, we demonstrate the effectiveness of our approach. MPOINSLI contains 167,743 Mapillary street-view images of 6,732 unique POIs, defined by the standard identifiers (Placekeys) which are further classified into 23 general functionalities categories (top-categories) and 67 more specific categories (sub-categories) of the POIs. MPOINSLI provides an open-source repository that contains metadata such as raw and post-processed camera-related parameters, the Harvesian distance between the camera and the POI's coordinates, and the intersection area. MPOINSLI could provide promising future applications for both smart cities and computer vision, including scene recognition across POI neighborhoods and fine-grained land-use classification.

_Key Words:_ Points-of-Interest, Neighborhood analythics, Smart-City Data, Mapillary Crowdsource street-level images, Safegraph

## Citing MPOINSLI
If you use MPOINSLI, please use the following BibTeX entry:
N. Zarbakhsh and G. McArdle, "Points-of-Interest from Mapillary Street-level Imagery: A Dataset For Neighborhood Analytics," 2023 IEEE 39th International Conference on Data Engineering Workshops (ICDEW), Anaheim, CA, USA, 2023, pp. 154-161, doi: 10.1109/ICDEW58674.2023.00030.
