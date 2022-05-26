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
    c = 5 # 5# 7 # 9
    r = 15 # 14 # 10 # 8
    points = 600 # 1200 # 15
    # save = '/Users/bernie/Desktop/course_material/FALL21/Skunk_dev/aggregate_exp/agg_friedman/agg_percen_fried/gpr_default_version'

    # save = f'CTR_EPSHN_run_rf_friedman_{c}_clusters_PGrOut2_{p}%ood_{points}_points_{id_tr}_id_tr_ratio_err_as_cali'
    # save = f'CTR_EPHN_run_rf_friedman_{c}_clusters_PGrOut2_{p}%ood_{points}_points_{id_tr}_id_tr_ratio_err_as_cali'

    # save = f'CTR_EPHN_run_rf_friedman_cluser_{c}cluser_{r}repeats_{points}_points_err_as_cali'
    # save = f'CTR_EPHN_maha_run_rf_friedman_cluser_{c}cluser_{r}repeats_{points}_points_err_as_cali'

    # save = f'NN_cosine_xent_1_60e_15b_0.11temp_fried_cluser_{c}cluser_{r}repeats_{points}_points_err_as_stdcal'

    # # sim Siamese
    # # save = f'Simsiam_fried_{c}cluser_{r}repeats_{points}_points_err_in_err_binned'
    # # save = f'Simsiam_fried_{c}cluser_{r}repeats_{points}_points_err_in_err_binned'
    # save = f'Simsiam_fried_PGrOut2_{p}%ood_{id_tr}_id_tr_ratio_err_in_err_binned'

    # sim Siamese
    # save = f'Transformer_fried_{c}cluser_{r}repeats_{points}_points_err_in_err_binned'
    # save = f'Transformer_fried_PGrOut2_{p}%ood_{id_tr}_id_tr_ratio_err_in_err_binned'
    # save = f'Transformer_fried_leave1out_err_in_err_binned'
    save = f'Transformer_150seq_fried_{c}cluser_{r}repeats_{points}_points_err_in_err_binned'

    uq_func = poly
    uq_coeffs_start = [0.0, 1.0, 0.1, 0.1, 0.1]

    # Load data
    data = load_data.friedman()
    df = data['frame']
    X = data['data']
    y = data['target']

    kmeans = KMeans(n_clusters=3, random_state=seed).fit(X)
    d = kmeans.predict(X)
    #


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

    # calibration_ctr.make_plots(save, points, 'stdcal', 'cosine_simsiam')
    # calibration_ctr.make_plots(save, points, 'stdcal', 'cosine')

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
