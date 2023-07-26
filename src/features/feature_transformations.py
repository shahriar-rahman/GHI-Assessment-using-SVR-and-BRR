import os
import sys
import polars as p
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
sys.path.append(os.path.abspath('../visualization'))
import visualize


class FeatureTransform:
    def __init__(self):
        self.df_ghi = p.read_csv('../../dataset/processed/ghi_processed.csv')
        self.vis = visualize.Visualize()

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

    def feature_transform(self):
        # Review DataFrame
        title = "Original Processed Data:"
        df_columns = self.df_ghi.columns
        self.debug_text(title, self.df_ghi.head(10))
        print(self.df_ghi.describe())

        super_title = "Original Scaling Distribution"
        self.vis.plot_multi_kde(self.df_ghi, df_columns, super_title)

        title = "Robust Scaling Data:"
        super_title = "Robust, Standard, and Min-Max Scaler distribution"
        # Robust Scaler
        scaler = preprocessing.RobustScaler()
        df_robust = scaler.fit_transform(self.df_ghi)
        df_robust = p.DataFrame(df_robust)
        df_robust.columns = df_columns
        self.debug_text(title, df_robust.describe())

        # Standard Scaler
        title = "Standard Scaling Data:"
        scaler = preprocessing.StandardScaler()
        df_standard = scaler.fit_transform(self.df_ghi)
        df_standard = p.DataFrame(df_standard)
        df_standard.columns = df_columns
        self.debug_text(title, df_standard.describe())

        # Min-Max Scaler
        title = "Min-Max Scaling Data:"
        scaler = preprocessing.MinMaxScaler()
        df_minmax = scaler.fit_transform(self.df_ghi)
        df_minmax = p.DataFrame(df_minmax)
        df_minmax.columns = df_columns
        self.debug_text(title, df_minmax.describe())

        self.vis.plot_compare_kde(df_robust, df_standard, df_minmax, df_columns, super_title)

        # Based on observations, Min-Max displays the most consistent deviation
        try:
            df_minmax.write_csv("../../dataset/transformed/ghi.csv", separator=",")

        except Exception as exc:
            self.debug_text('! Exception encountered', exc)

        else:
            print("Data Storage Successful!")


if __name__ == "__main__":
    main = FeatureTransform()
    main.feature_transform()