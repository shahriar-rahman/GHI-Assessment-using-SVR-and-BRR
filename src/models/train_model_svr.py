import os
import sys
import numpy as np
import polars as pl
import joblib as jb
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
sys.path.append(os.path.abspath('../visualization'))
import visualize
path = "../../dataset/transformed/ghi.csv"
test_size = 0.25
random_state = 48


class TrainModel:
    def __init__(self):
        self.df_ghi = pl.read_csv(path)
        self.visualize = visualize.Visualize()

    @staticmethod
    def debug_text(title, task):
        print('\n')
        print("=" * 150)
        print('◘ ', title)

        try:
            print(task)

        except Exception as exc:
            print("! ", exc)

        finally:
            print("=" * 150)

    def train_model(self):
        # Construct the Predictor and Response Variables
        self.debug_text("Dataframe Columns:", self.df_ghi.columns)
        self.df_ghi = self.df_ghi.drop('hour')

        x_data = self.df_ghi.clone()
        x_data = x_data.drop('ghi')
        self.debug_text("Data X:", x_data)

        y_data = self.df_ghi.select('ghi')
        self.debug_text("Data Y:", y_data)

        # Partition the Structured Data
        train_x, test_x, train_y, test_y = train_test_split(x_data, y_data, test_size=test_size,
                                                            random_state=random_state)
        self.debug_text("Shape of Training set:", train_x.shape)
        self.debug_text("Shape of Testing set:", test_x.shape)
        self.store_test(test_x, test_y)

        # Address the multicollinearity issue using PCA
        # Scale Predictor Variables
        pca = PCA()
        pca_x_train = pca.fit_transform(scale(train_x))
        flat_y_train = np.ravel(train_y)

        # Define Cross-Validation Method
        cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

        svr = SVR()
        mse_list = []

        # Calculate MSE using cross-validation, incrementing component gradually
        current_row = 0
        current_column = 0
        row_max = 3
        col_max = 4
        fig, ax = plt.subplots(row_max, col_max, figsize=(19, 23))

        for predictor in np.arange(1, pca_x_train.shape[1]+1):  #
            score = -1 * model_selection.cross_val_score(svr, pca_x_train[:, :predictor], flat_y_train, cv=cv,
                                                         scoring='neg_mean_squared_error').mean()
            mse_list.append(score)

            predict_y = cross_val_predict(svr, pca_x_train[:, :predictor], flat_y_train, cv=10)
            if current_row < row_max and current_column < col_max:
                ax[current_row][current_column].scatter(predict_y, flat_y_train, color='maroon')
                ax[current_row][current_column].set_title(str(predictor) + ' Predictors', fontsize=12,
                                                          fontweight='bold')

            current_column += 1
            if current_column > col_max-1:
                current_row += 1
                current_column = 0

        fig.delaxes(ax[2, 2])
        fig.delaxes(ax[2, 3])
        fig.suptitle("CV Residual Comparison using SVR", fontsize=18, fontweight='bold')
        plt.show()

        # Plot Cross-Validation assessment
        title = "CV Score Comparison on SVR using PCA"
        x_label = "Number of Principal Components"
        y_label = "Mean Squared Error"
        self.debug_text("MSE Scores:", mse_list)
        self.visualize.plot_graph(mse_list, title, x_label, y_label)

        # Select the most Optimum Components to rescale data
        pca = PCA(n_components=4)
        pca_x_train = pca.fit_transform(train_x)

        # Grid CV Search for SVR
        kernel = ['linear', 'rbf', 'sigmoid']
        c = [1.0, 1.5, 2.0]
        epsilon = [0.1, 0.2, 0.3]
        svr_parm = {'kernel': kernel, 'C': c, 'epsilon': epsilon}

        svr_grid = GridSearchCV(svr, svr_parm, scoring='neg_mean_squared_error', cv=10)
        grid_search = svr_grid
        grid_search.fit(pca_x_train, flat_y_train)

        # Hyperparameter diagnostics
        self.debug_text("Ideal parameters: ", grid_search.best_params_)
        self.debug_text("Ideal Score: ", grid_search.best_score_)
        ideal_param_list = [grid_search.best_params_['kernel'], grid_search.best_params_['C'],
                            grid_search.best_params_['epsilon']]
        self.debug_text("Ideal Kernels, C, and Epsilon:", ideal_param_list)

        # Train the SVR model using the diagnosed Hyperparameters
        svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        svr.fit(pca_x_train, flat_y_train)
        predict_y = svr.predict(pca_x_train)

        # Loss Functions
        loss_mse = mse(flat_y_train, predict_y, squared=True)
        loss_r_mse = mse(flat_y_train, predict_y, squared=False)
        loss_r2 = r2_score(flat_y_train, predict_y)
        loss_mae = mae(flat_y_train, predict_y)

        self.debug_text("MSE (Training data):", loss_mse)
        self.debug_text("RMSE (Training data):", loss_r_mse)
        self.debug_text("R-Squared (Training data):", loss_r2)
        self.debug_text("MAE (Training data):", loss_mae)

        title = "SVR Residual Loss (Training data)"
        self.visualize.plot_residual(flat_y_train, predict_y, title, "Actual Values", "Predicted Values")

        bins = 10
        x_label = 'Errors'
        y_label = 'Frequency'
        error_list = predict_y - flat_y_train
        title = "SVR Prediction error distribution (Training data)"
        self.visualize.plot_dist(error_list, bins, title, x_label, y_label)

        # Store Model
        try:
            jb.dump(svr, f'../../models/svr.pkl')

        except Exception as exc:
            self.debug_text("! Exception encountered", exc)

        else:
            self.debug_text("• Model saved successfully", '')

    @staticmethod
    def store_test(test_x, test_y):
        try:
            test_x.write_csv("../../dataset/tests/ghi_test_x.csv", separator=",")
            test_y.write_csv("../../dataset/tests/ghi_test_y.csv", separator=",")

        except Exception as exc:
            print("! ", exc)

        else:
            print('\nTest X shape: ', test_x.shape)
            print('Test Y shape: ', test_y.shape)
            print("• Test Data write successful!")


if __name__ == "__main__":
    main = TrainModel()
    main.train_model()
