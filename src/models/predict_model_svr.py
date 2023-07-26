import os
import sys
import numpy as np
import polars as pl
import joblib as jb
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
sys.path.append(os.path.abspath('../visualization'))
import visualize
path_test = ['../../dataset/tests/ghi_test_x.csv', '../../dataset/tests/ghi_test_y.csv']


class PredictModel:
    def __init__(self):
        self.visualize = visualize.Visualize()
        self.test_x = pl.read_csv(path_test[0])
        self.test_y = pl.read_csv(path_test[1])
        self.svr = jb.load(f'../../models/svr.pkl')

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

    def predict_model(self):
        # Apply PCA on the data and Load Model
        pca = PCA(n_components=4)
        pca_x_test = pca.fit_transform(self.test_x)
        flat_y_test = np.ravel(self.test_y)
        predict_y = self.svr.predict(pca_x_test)

        # Loss Functions
        loss_mse = mse(flat_y_test, predict_y, squared=True)
        loss_r_mse = mse(flat_y_test, predict_y, squared=False)
        loss_r2 = r2_score(flat_y_test, predict_y)
        loss_mae = mae(flat_y_test, predict_y)

        self.debug_text("MSE (Test data):", loss_mse)
        self.debug_text("RMSE (Test data):", loss_r_mse)
        self.debug_text("R-Squared (Test data):", loss_r2)
        self.debug_text("MAE (Test data):", loss_mae)

        title = "SVR Residual Loss (Test data)"
        self.visualize.plot_residual(flat_y_test, predict_y, title, "Actual Values", "Predicted Values")

        bins = 10
        x_label = 'Errors'
        y_label = 'Frequency'
        error_list = predict_y - flat_y_test
        title = "SVR Prediction error distribution (Test data)"
        self.visualize.plot_dist(error_list, bins, title, x_label, y_label)


if __name__ == "__main__":
    main = PredictModel()
    main.predict_model()
