# A Benchmark Study for the Assessment of Global Horizontal Irradiation using SVR and BRR
**Global Horizontal Irradiance (GHI)** Prediction using the concepts of *Principal Component Analysis (PCA)* to reduce the dimensionality of the data, *Cross-Validation (CV)* to evaluate a model's interpretability, based on *Support Vector Regression (SVR)*, and *Bayesian Ridge Regression (BRR)*.

</br>

![alt text](https://github.com/shahriar-rahman/GHI-Assessment-using-SVR-and-BRR/blob/main/img/solarA.jpg)

</br>

### ◘ Introduction
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


### *ReadMe Construction in Progress...*

</br>
