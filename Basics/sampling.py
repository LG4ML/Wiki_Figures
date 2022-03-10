import numpy as np
import pandas as pd
from typing import Tuple, List
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.pipeline import make_pipeline


def generate_cluster_from_center(center_point: Tuple[float, float], n_points: int = 50,
                                 norm_scale: float = 0.5) -> List[Tuple[float, float]]:
    return [(center_point[0] + np.random.normal(0, norm_scale), center_point[1] + np.random.normal(0, norm_scale))
            for _ in range(n_points)]


def plot_dataset(data, title):
    fig, axs = plt.subplots()
    fig.set_size_inches(12, 8)
    axs.scatter(data[data.Label == 'Fraud'].x1, data.x2[data.Label == 'Fraud'], color='orange',
                label='Betrug')
    axs.scatter(data[data.Label != 'Fraud'].x1, data.x2[data.Label != 'Fraud'], color='lightblue',
                label='Kein Betrug')
    axs.set(title=title, xlabel='x1', ylabel='x2')
    axs.legend(['Betrug', 'Kein Betrug'])
    fig.savefig(f'../figures/Sampling_{title}.png')
    plt.show()


# Create imbalanced dataset
np.random.seed(42)
minority_class = generate_cluster_from_center((2.5, 3), norm_scale=1.0, n_points=20)
majority_class = generate_cluster_from_center((1.5, 1.5), norm_scale=0.5, n_points=180)
label = ['Fraud'] * len(minority_class) + ['No Fraud'] * len(majority_class)
x1, x2 = [p[0] for p in minority_class + majority_class], [p[1] for p in minority_class + majority_class]
dataset = pd.DataFrame(data={'x1': x1, 'x2': x2, 'Label': label})
plot_dataset(dataset, 'Originaler Datensatz')

# Apply Undersampling
X_under, y_under = RandomUnderSampler(sampling_strategy='auto').fit_resample(dataset.drop(columns=['Label']), dataset.Label)
df_under = pd.DataFrame(X_under, columns=['x1', 'x2'])
df_under['Label'] = y_under
plot_dataset(df_under, 'Undersampling')

# Apply Oversampling
X_over, y_over = RandomOverSampler(sampling_strategy='auto').fit_resample(dataset.drop(columns=['Label']), dataset.Label)
df_over = pd.DataFrame(X_over, columns=['x1', 'x2'])
df_over['Label'] = y_over
plot_dataset(df_over, 'Oversampling')

# Apply SMOTE
X_smote, y_smote = SMOTE(sampling_strategy='auto').fit_resample(dataset.drop(columns=['Label']), dataset.Label)
df_smote = pd.DataFrame(X_smote, columns=['x1', 'x2'])
df_smote['Label'] = y_smote
plot_dataset(df_smote, 'SMOTE')
