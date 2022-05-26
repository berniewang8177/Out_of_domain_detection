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

    p = 0.15
    id_tr = 0.2

    points =  int(250)#  15

    c = 5 # 15 # 3 # 5 # 9
    r = 15  # 5 #  25 # 15 # 8

    # save = '/Users/bernie/Desktop/course_material/FALL21/Skunk_dev/aggregate_exp/agg_supercond/agg_Percen_supercond/gpr_default_version'
    # save = f'CTR_SCT_rf_diffusion_PGOut_{p}%ood_{points}_points_{id_tr}_id_tr_ratio'
    # save = f'CTR_EPSHN_rf_diffusion_PGOut_{p}%ood_{points}_points_{id_tr}_id_tr_ratio_err_as_ares'
    # save = f'CTR_EPHN_att_rf_diffusion_PGOut_{p}%ood_{points}_points_{id_tr}_id_tr_ratio_err_as_cali'

    #save = f'TEST_att_CTR_EPHN_run_rf_diffu_cluser_{c}cluser_{r}repeats_{points}_points_err_as_cali'
    # save = f'CTR_EPHN_maha_run_rf_diffu_cluser_{c}cluser_{r}repeats_{points}_points_err_as_cali_8b_60e'
    # save = f'CTR_EPHN_cosine_run_rf_diffu_PGOUT__{p}%ood_{points}_points_{id_tr}id_tr_ratio_err_as_cali_20b_50e'
    #

    # NN experiment
    # save = f'NN_maha_EPHN_diffu_cluser_{c}cluser_{r}repeats_{points}_points_err_as_cali'
    # save = f'NN_cosine_EPHN_diffu_cluser_{c}cluser_{r}repeats_{points}_points_err_as_cali'
    # save = f'Simsiam_diffu_cluser_{c}cluser_{r}repeats_{points}_points_err_in_err_binned'
    # save = f'NN_maha_EPHN_diffu_cluser_{c}cluser_{r}repeats_{points}_points_err_as_std_cal'

    # Simsiam experiment
    # save = f'Simsiam_diffu_PGOUT_{p}%ood_{points}_points_{id_tr}id_tr_ratio_err_in_err_binned'
    # save = f'Simsiam_diffu_leave1chemical_20repeats_err_in_err_binned'
    # save = f'Simsiam_diffu_{c}cluser_{r}repeats_{points}_points_err_in_err_binned'

    # Transformer experiment
    # save = f'Transformer_diffu_PGOUT_{p}%ood_{points}_points_{id_tr}id_tr_ratio_err_in_err_binned'
    # save = f'Transformer_diffu_leave1chemical_20repeats_err_in_err_binned'
    # save = f'Transformer_diffu_{c}cluser_{r}repeats_{points}_points_err_in_err_binned'
    save = f'Transformer_100seq_3depth8heads_diffu_{c}cluser_{r}repeats_{points}_points_err_in_err_binned'
    # #
    uq_func = poly
    uq_coeffs_start = [0.0, 1.0, 0.1, 0.1, 0.1]
    #
    # Load data
    data = load_data.diffusion()
    df = data['frame']
    X = data['data']
    y = data['target']
    d = data['class_name']
    # #
    # Splitters
    top_split = None
    mid_split = splitters.RepeatedClusterSplit(
                                               KMeans,
                                               n_repeats=r,
                                               n_clusters=c
                                               )
    # mid_split = splitters.PercentageGroupOut(  n_repeats=20,
    #                                             groups = d,
    #                                             percentage= p,
    #                                             id_tr_ratio = id_tr) # ensure id:train <= 1:5
    #
    # # mid_split = splitters.BootstrappedLeaveOneGroupOut( n_repeats=20, groups = d)
    bot_split = RepeatedKFold(n_splits=5, n_repeats=1)
    # # #
    # ML setup
    scale = StandardScaler()
    selector = feature_selectors.no_selection()

    # Random for:est regression
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
                            dataset_name = 'diffu'
                            )

    splits.assess_domain()  # Do ML
    splits.aggregate()  # combine all of the ml data
    statistics.folds(save)  # Gather statistics from data


    calibration.make_plots(save, points, 'stdcal', 'cosine_transformer')
    calibration.make_plots(save, points, 'stdcal', 'cosine')

    # calibration_ctr.make_plots(save, points, 'stdcal', 'cosine_transformer')


    # calibration_ctr.make_plots(save, points, 'stdcal', 'cosine_simsiam')
    # calibration_ctr.make_plots(save, points, 'stdcal', 'cosine')

    calibration.make_plots(save, points, 'stdcal', 'mahalanobis')
    calibration.make_plots(save, points, 'stdcal', 'mahalanobis_transformer')

    # calibration_ctr.make_plots(save, points, 'stdcal', 'mahalanobis')
    # calibration_ctr.make_plots(save, points, 'stdcal', 'mahalanobis_transformer')
    # #
    # calibration.make_plots(save, points, 'stdcal', 'oneClassSVM')
    # calibration.make_plots(save, points, 'stdcal', 'oneClassSVM_simsiam')
    #
    # calibration.make_plots(save, points, 'stdcal', 'lof')
    # calibration.make_plots(save, points, 'stdcal', 'lof_simsiam')

    # calibration_ctr.make_plots(save, points, 'stdcal', 'maha_nn')
    #
    # calibration_ctr.make_plots(save, points, 'stdcal', 'attention_metric')
    # calibration_ctr.make_plots(save, points, 'stdcal', 'attention_metric_ctr')

    # calibration_ctr.make_plots(save, points, 'stdcal', 'gpr_std')
    # calibration_ctr.make_plots(save, points, 'stdcal', 'oneClassSVM')


if __name__ == '__main__':
    main()
