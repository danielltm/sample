from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition.pca import PCA

def get_standar_scale_with_pca_etc() -> Pipeline:
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('reduce_dim', PCA())
    ])
    return pipe

