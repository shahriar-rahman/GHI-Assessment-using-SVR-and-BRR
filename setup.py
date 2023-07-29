from setuptools import find_packages, setup

setup(
    name='ghi_assessment_package',
    packages=find_packages(),
    version='1.0.0',
    description='Global Horizontal Irradiance Analysis using Support Vector Regression and Bayesian Ridge Regression.',
    author='Shahriar Rahman',
    license='MIT License',
    author_email='shahriarrahman1101@gmail.com',
    url='https://github.com/shahriar-rahman/GHI-Assessment-using-SVR-and-BRR',
    python_requires='>=3.11, <4',
    install_requires=[
        'matplotlib~=3.7.1',
        'polars~=0.18.8',
        'seaborn~=0.12.2',
        'scikit-learn~=1.2.2',
        'missingno~=0.5.2',
        'numpy~=1.24.2',
        'joblib~=1.2.0',
        'setuptools~=65.5.1',
    ],
)
