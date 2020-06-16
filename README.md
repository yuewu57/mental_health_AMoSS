# mental_health_AMoSS
This repository stores the resources to rework on ASRM/QIDS data collected from AMoSS study. <sup>1<sup>

Basically, the analysis is signature-based. <sup>2<sup> 
  
All the results are contained in preprint **Deriving information from missing data: implications for mood prediction**.
  
Getting Started
---------------

Once the repo has been cloned locally, setup a python environment with ``python==3.6`` and run ``pip install -r requirements.txt``.

As the data were collected pre-GDPR and contained sensitive personal data, it cannot be placed into a publicly accessible repository. Requests for access to the data can be made to the data controller G.M.Goodwin. 

Patient Group
---------------
| Diagnosis Group   |  Class|
|------------|--------|
|Borderline|0|
|Healthy|1|
|Bipolar|2|


XX.py structure
---------------
| File    | Task| Section in manuscript|
|----------|------------|--------|
|``data_cleaning.py``| cleaning dataset|Sec 2.1|
|``data_transforms.py``| handling data|Sec 2.2|
|``classifiers.py``| classification |Sec 2.3|
|``ROC_functions.py``| ROC plots |Sec 2.3|
|``spectrum_functions.py``| spectrum plots for classification|Sec 2.3.1|
|``prediction_functions.py``| prediction for classification |Sec 2.4.1 and Sec 2.4.3|
|``prediction_spectrum_functions.py``| spectrum plots for prediction |Sec 2.4.2|

Notebooks for results
---------------
The results have been collected and shown in different jupyter notebooks.

| File    | Task| Table/Figure in manuscript|
|----------|------------|--------|
|``classification_task.ipynb``| classification|Table 2 and Figure S1|
|``ROC_classification.ipynb``| ROC plots |Figure S2|
|``loo_cv_spectrum_classification.ipynb``| spectrum plots for classification|Figure S3|
|``prediction_tasks.ipynb``| prediction |Table 3-5|
|``loo_cv_spectrum_prediction.ipynb``| spectrum plots for prediction|Figure S5|

References
---------------
  1. Tsanas A, Saunders KE, Bilderbeck AC, Palmius N, Osipov M, Clifford GD, Goodwin GΜ, De Vos M. Daily longitudinal self-monitoring of mood variability in bipolar disorder and borderline personality disorder. *Journal of affective disorders*. 2016 Nov 15;205:225-33. doi:10.1016/j.jad.2016.06.065
 
  2. Lyons T. Rough paths, signatures and the modelling of functions on streams. *arXiv preprint arXiv:1405.4537*. 2014 May 18.
