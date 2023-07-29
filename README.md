# A Benchmark Study for the Assessment of Global Horizontal Irradiation using SVR and BRR
**Global Horizontal Irradiance (GHI)** Prediction using the concepts of *Principal Component Analysis (PCA)* to reduce the dimensionality of the data, *Cross-Validation (CV)* to evaluate a model's interpretability, based on *Support Vector Regression (SVR)*, and *Bayesian Ridge Regression (BRR)*.

</br>

![alt text](https://github.com/shahriar-rahman/GHI-Assessment-using-SVR-and-BRR/blob/main/img/solarA.jpg)

</br></br>

* [Introduction](#-introduction)
* [Important Notes](#important-notes)
	* [Session Table Collation](#session-table-collation)
* [Usage](#usage)
	* [Use an existing MySQL connection or pool](#use-an-existing-mysql-connection-or-pool)
	* [Closing the session store](#closing-the-session-store)
	* [Options](#options)
		* [Custom database table schema](#custom-database-table-schema)
	* [Debugging](#debugging)
* [Contributing](#contributing)
	* [Configure Local Environment](#configure-local-environment)
	* [Tests](#tests)
* [Changelog](#changelog)
* [License](#license)


</br></br>

### • Introduction
Solar irradiance is the amount of power per unit area (surface power density) that is received from the Sun in the form of electromagnetic radiation in the wavelength range of the measuring device. 
The radiation reaching the earth's surface can be represented in a number of different ways. Global Horizontal Irradiance (GHI) is the total amount of shortwave radiation received from above by a surface horizontal to the ground. It is important to point out that the sum can be correctly acquired after accounting for the solar zenith angle of the Sun z and diffuse horizontal irradiance:

</br>

```
GHI = DHI + DNI × cos ⁡ ( z ) 
```
</br>

The Irradiance can be observed and measured at the Earth's surface after atmospheric absorption and scattering and therefore, is a function of distance from the Sun, the solar cycle, and cross-cycle changes in space. The predominant advantages for the study and measuring the irradiance are the prediction of energy generation from solar power plants to better strategize the power consumption, the heating and cooling loads of buildings, climate modeling and weather forecasting, passive daytime radiative cooling applications, and space travel.

</br></br>

### ◘ Objective
The primary initiative of this research is to:
* Create a benchmark study of how different algorithms can affect the overall generalizability of the model.
* Apply various transformation techniques to increase the effectiveness of the data.
* Tune the Hyperparameters to ensure the correct training procedure.
* Conduct a thorough analysis in order to estimate the Global Horizontal Irradiance on the earth's atmosphere using the NSRDB weather data.
* Compare and contrast the results from the models and reach a conclusion on which model to select based on effectiveness and performance.

</br></br>

### ◘ Approach
* Process and convert the modified data to increase the interpretability of the models.
* Correctly generalize a decision boundary for the Predictor and the Response Variables.
* Partition of the data to devise Training and Test samples.
* Address the multicollinearity issue using PCA.
* Identify the correct components using CV on PCA
* Analyze the correct Hyperparameters for the aforementioned models.
* Appy all the information to initiate the training procedure of the models
* Evaluate the model performance and pinpoint the strengths and weaknesses.

</br>

### ◘ Feature Transformations Flowchart
![alt text](https://github.com/shahriar-rahman/GHI-Assessment-using-SVR-and-BRR/blob/main/img/Flow1.JPG)

</br>


### Principal Component Analysis (PCA)
In this case study, the acquired dataset contains attributes that can be considered correlated despite some inconsistencies that primarily stem due to seasonal alterations and other variables not present in the weather data. The multiple explanatory variables in the multiple regression model being highly linearly related, can cause a multicollinearity issue. This would lead to the model finding it challenging to interpret the coefficients, resulting in reduced power of the model to identify independent variables that are statistically significant, which would be a serious problem. To address this issue,PCA is applied and combined with the Cross Validation technique to find an average score for all the predictor variables involved.

</br>

PCA is commonly applied in exploratory data analysis and for making predictive models and is mostly used for dimensionality reduction by projecting each data point onto only the first few principal components to obtain lower-dimensional data while preserving as much of the data's variation as possible. The principal components of a collection of points in a real coordinate space are a sequence of *p-unit* vectors, where the *i*th vector is the direction of a line that best fits the data while being orthogonal to the first *i-1* vectors. Therefore, the PCA technique is utilized for analyzing large datasets containing many dimensions/features per observation, increasing the interpretability of the data by reducing the dimensionality of a dataset while preserving the maximum amount of information, and enabling the visualization of multidimensional data. 

</br>

### ◘ Model Training Flowchart
![alt text](https://github.com/shahriar-rahman/GHI-Assessment-using-SVR-and-BRR/blob/main/img/Flow2.JPG)

</br>

### ◘ Support Vector Regression (SVR)
SVR is a regression function that is generalized by Support Vector Machines - a machine learning model used for data classification on continuous data. In simple regression, the idea is to minimize the error rate while in SVR the idea is to fit the error inside a certain threshold which means, the work of SVR is to approximate the best value within a given margin by finding the optimal decision boundary that maximally separates different points. Such regression line is represented as:

</br>

```
y^ = WTx+b
Where y^,WT,x, and b  are defined as:
y^: Estimated value
WT: Weight vector
x: Explanatory variable
b: Bias term
```

</br></br>

### ◘ Bayesian Ridge Regression (BRR)
BRR follows Bayes' theorem which describes the conditional probability of an event based on prior knowledge of conditions that might be related to the event. Since Bayesian statistics treat probability as a degree of belief, Bayes' theorem can directly assign a probability distribution that quantifies the belief to the parameter or set of parameters. As a result, BRR allows a natural mechanism to survive insufficient data or poorly distributed data by formulating linear regression using probability distribution rather than point estimates. The output or response ‘y’ is assumed to be drawn from a probability distribution rather than estimation as a single value.  Mathematically, to obtain a fully probabilistic model the response y is assumed to be Gaussian distributed around Xw as follows:

</br>

```
p(y | X, w, a) = N(y | X, w, a)
where w is the weight
and a is the penalty coefficient
```

</br>

### ◘ Model Testing Flowchart
![alt text](https://github.com/shahriar-rahman/GHI-Assessment-using-SVR-and-BRR/blob/main/img/Flow3.JPG)

</br></br>

### ◘ Model Evaluation
| **Measurement** | **SVR Training** | **BRR Training** | **SVR Test** | **BRR Test** |
| -- | -- | -- | -- | -- |
| MSE | 0.0015 | 7.27e-06 | 0.0015 | 6.94e-3 |   
| MAE | 0.0318 | 0.0014 | 0.0320 | 0.0622 |
| RMSE | 0.0392 | 0.0026 | 0.0387 | 0.0834 |
| R-Squared | 0.9751 | 0.9998 | 0.9780 | 0.8977 |


<br/><br/>

### ◘ Project Organization
------------
    ├─-- LICENSE
    |
    ├─-- README.md              # The top-level README for developers using this project
    |
    ├─-- dataset                # Different types of data derived from the original raw dataset
    |    └──  processed        
    |    └──  raw
    |    └──  tests
    |    └──  transformed
    |    
    ├─ graphs                    # Generated graphics and figures obtained from visualization.py
    |
    |
    ├─-- img                    # Project related files
    |
    |
    ├─-- models                 # Trained and serialized models for future model predictions  
    |    └── svr.pkl
    |    └── brr.pkl
    |
    |
    ├─-- src                    # Source code for use in this research
    |   └───-- __init__.py    
    |   |
    |   |
    |   ├─-- features            # Scripts to modify the raw data into different types of features for modeling
    |   |   └── feature_transformations.py
    |   |
    |   |
    |   ├─-- models                # Contains py filess for inspecting hyperparameters, training, and using trained models to make predictions         
    |   |   └─── predict_model_brr.py
    |   |   └─── predict_model_svr.py
    |   |   └─── train_model_brr.py
    |   |   └─── train_model_svr.py
    |   |
    |   |
    |   └───-- visualization        # Construct exploratory and result oriented visualizations to identify and reaffirm patterns
    |       └───-- visualize.py
    |
    |
    ├─-- requirements.txt       # The requirements file for reproducing the analysis environments
    |                         
    |
    ├─-- setup.py               # makes project pip installable, so that src can be imported
    |
    ├─
--------

<br/><br/>

### ◘ Libraries & Technologies utilized
* matplotlib~=3.7.1
* polars~=0.18.8
* seaborn~=0.12.2
* scikit-learn~=1.2.2
* missingno~=0.5.2
* numpy~=1.24.2
* joblib~=1.2.0
* setuptools~=65.5.1


<br/><br/>


<br/><br/>

### ◘ Module Installation (setup.py)
1. To use the *setup.py* file in Python, the first objective is to have the *setuptools* module installed. It can be accomplished by running the following command:
```
pip install setuptools                                     
```
2. Once the setuptools module is installed, use the setup.py file to build and distribute the Python package by running the following command:
```
python setup.py sdist bdist_wheel
```
3. In order to install the my_package package, run the following command:
```
pip install my_package                                 
```
4. This command will install all the requirements:
```
pip install .                                 
```
5. This will install the my_package package and any of its dependencies that are not already installed on your system. Once the package is installed, you can use it in your Python programs by importing it like any other module. For example:
```
import my_package                                
```

<br/><br/>

### ◘ Python Library Installation (using pip)
In order to *install* the required packages on the local machine, Open pip and run the following commands separately:
```
> pip install setuptools                    

> pip install polars                                                          

> pip install scikit-learn                                      

> pip install seaborn

> pip install matplotlib

> pip install missingno

> pip install numpy

> pip install joblib                                  
```

<br/><br/>

### ◘ Supplementary Resources
For more details, visit the following links:
* https://pypi.org/project/setuptools/
* https://pypi.org/project/polars/
* https://pypi.org/project/scikit-learn/
* https://pypi.org/project/seaborn/
* https://pypi.org/project/matplotlib/
* https://pypi.org/project/missingno/
* https://pypi.org/project/numpy/
* https://pypi.org/project/joblib/

<br/><br/>

### ◘ MIT License
Copyright (c) 2023 Shahriar Rahman

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

<br/>

===========================================================================

