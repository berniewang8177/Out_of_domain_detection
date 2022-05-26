from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans

from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from mad.ml import splitters, feature_selectors, domain_transformer
from mad.datasets import load_data, statistics
from mad.plots import parity, calibration, calibration_ctr
from mad.functions import poly

import numpy as np

def main():
    '''
    Test ml workflow
    '''

    seed = 14987
    p = 0.5
    id_tr = 0.2
    c = 3
    r = 8
    points = 4250 # 1200 # 15


    # sim Siamese
    save = f'Transformer_supercond__{c}cluser_{r}repeats_err_in_err_binned'
    # save = f'Transformer_supercond__PGrOut2_{p}%ood_{id_tr}_id_tr_ratio_err_in_err_binned'

    # save = f'Transformer_supercond_leave1ChemiGroup_20repeats_err_in_err_binned'

    uq_func = poly
    uq_coeffs_start = [0.0, 1.0, 0.1, 0.1, 0.1]

    # Load data
    data = load_data.super_cond()
    df = data['frame']
    X = data['data']
    y = data['target']
    d = data['class_name']
    # kmeans = KMeans(n_clusters=3, random_state=seed).fit(X)
    # d = kmeans.predict(X)

    # Splitters
    top_split = None
    mid_split = splitters.RepeatedClusterSplit(
                                               KMeans,
                                               n_repeats=r,
                                               n_clusters=c
                                               )
    # mid_split = splitters.PercentageGroupOut2(  n_repeats=20,
    #                                             groups = d,
    #                                             percentage= p,
    #                                             id_tr_ratio = id_tr)
    #
    # mid_split = splitters.BootstrappedLeaveOneGroupOut( n_repeats=20, groups = d)
    bot_split = RepeatedKFold(n_splits=5, n_repeats=1)

    # ML setup
    scale = StandardScaler()
    selector = feature_selectors.no_selection()

    # Random forest regression
    grid = {}
    model = RandomForestRegressor()
    grid['model__n_estimators'] = [100]
    grid['model__max_features'] = [None]
    grid['model__max_depth'] = [None]
    pipe = Pipeline(steps=[
                           ('scaler', scale),
                           ('select', selector),
                           ('model', model)
                           ])
    rf = GridSearchCV(pipe, grid, cv=bot_split)

    # Evaluate

    splits = domain_transformer.builder(
                            rf,
                            X,
                            y,
                            d,
                            top_split,
                            mid_split,
                            save,
                            seed=seed,
                            uq_func=uq_func,
                            uq_coeffs_start=uq_coeffs_start,
                            dataset_name = 'fried'
                            )

    splits.assess_domain()  # Do ML
    splits.aggregate()  # combine all of the ml data
    statistics.folds(save)  # Gather statistics from data


    calibration.make_plots(save, points, 'stdcal', 'cosine_transformer')
    calibration.make_plots(save, points, 'stdcal', 'cosine')


    calibration.make_plots(save, points, 'stdcal', 'mahalanobis')
    calibration.make_plots(save, points, 'stdcal', 'mahalanobis_transformer')

    # calibration_ctr.make_plots(save, points, 'stdcal', 'gpr_std')
    # calibration_ctr.make_plots(save, points, 'stdcal', 'oneClassSVM')

    # calibration.make_plots(save, points, 'stdcal', 'mahalanobis')
    # calibration.make_plots(save, points, 'stdcal', 'attention_metric')
    # calibration.make_plots(save, points, 'stdcal', 'attention_metric_ts_ss')
    # calibration.make_plots(save, points, 'stdcal', 'oneClassSVM')


if __name__ == '__main__':
    main()
