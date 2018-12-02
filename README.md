
### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

There code only requires the standard installation of Anaconda Python.  The following libraries need to be installed for this project to work:
- psycopg2
- sqlalchemy
- pandas
- xgboost
- sklearn
- re
- matplotlib
- numpy

The MIMIC-III dataset has been provided by MIT Laboratory for Computational Physiology.  The data access is only avaiable after approval.  The request for data access is available via this link: [https://mimic.physionet.org/gettingstarted/access/](https://mimic.physionet.org/gettingstarted/access/)

The data was loaded into a local PostgresSQL instance.  The instructions on how to setup the PostgresSQL instances is available via this link: [https://github.com/MIT-LCP/mimic-code/tree/master/buildmimic](https://github.com/MIT-LCP/mimic-code/tree/master/buildmimic)

## Project Motivation<a name="motivation"></a>

The purpose of this project is to evaluate hospital admisisons data and to develop a model that can predict 30-day readmissions.  The hope is that a model can be used to identify patients will be readmitted, so that the hospital can take the necessary steps to prevent the readmissions.  By doing so, the hospital can ensure that patients received the appropriate care and time and money are not wasted on unnecessary readmissions.

## File Descriptions <a name="files"></a>

There are 4 files.  There is 1 Jupyter Notebook that contains the data analysis and data modeling done for this project.  

There are 3 SQL files that are used to query the local PostgresSQL instance that contains the MIMIC-III database.

Github Link: [https://github.com/dkim319/30DayReadmissionsPredictions](https://github.com/dkim319/30DayReadmissionsPredictions)

## Results<a name="results"></a>

The findings are documented in this Medium post. 

Link: [https://medium.com/@dkim319/is-it-possible-to-predict-30-day-readmissions-8d61b106442b](https://medium.com/@dkim319/is-it-possible-to-predict-30-day-readmissions-8d61b106442b)

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

The MIMIC-III data was accessible through the MIT Laboratory for Computational Physiology.  Data access requires approval. 

Link: [https://mimic.physionet.org/gettingstarted/access/](https://mimic.physionet.org/gettingstarted/access/)

Citations
MIMIC-III, a freely accessible critical care database. Johnson AEW, Pollard TJ, Shen L, Lehman L, Feng M, 
Ghassemi M, Moody B, Szolovits P, Celi LA, and Mark RG. Scientific Data (2016). DOI: 10.1038/sdata.2016.35. 
Available at: http://www.nature.com/articles/sdata201635

Pollard, T. J. & Johnson, A. E. W. The MIMIC-III Clinical Database http://dx.doi.org/10.13026/C2XW26 (2016).

Physiobank, physiotoolkit, and physionet components of a new research resource for complex physiologic signals. 
Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov P, Mark RG, Mietus JE, Moody GB, Peng C, and Stanley HE. 
Circulation. 101(23), pe215–e220. 2000.

Johnson, Alistair EW, David J. Stone, Leo A. Celi, and Tom J. Pollard. “The MIMIC Code Repository: enabling 
reproducibility in critical care research.” Journal of the American Medical Informatics Association (2017): ocx084.
