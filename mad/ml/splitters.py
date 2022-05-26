from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.model_selection import train_test_split
from sklearn.cluster import estimate_bandwidth
from sklearn.neighbors import KernelDensity
from sklearn.utils import resample

import pandas as pd
import numpy as np
import random


class NoSplit:
    '''
    Class to not split data
    '''

    def get_n_splits(self):
        '''
        Should only have one split.
        '''

        return 1

    def split(self, X, y=None, groups=None):
        '''
        The training and testing indexes are the same.
        '''

        indx = range(X.shape[0])

        yield indx, indx


class BootstrappedLeaveOneGroupOut:
    '''
    Custom splitting class which with every iteration of n_repeats it will
    bootstrap the dataset with replacement and leave every group out once
    with a given class column.
    '''

    def __init__(self, n_repeats, groups, *args, **kwargs):
        '''
        inputs:
            n_repeats = The number of times to apply splitting.
            groups =  np.array of group classes for the dataset.
        '''

        self.groups = groups
        self.n_repeats = n_repeats

    def get_n_splits(self, X=None, y=None, groups=None):
        '''
        A method to return the O(N) number of splits.
        '''

        self.groups = groups
        self.n_splits = self.n_repeats*len(set(groups))

        return self.n_splits

    def split(self, X=None, y=None, groups=None):
        '''
        For every iteration, bootstrap the original dataset, and leave
        every group out as the testing set one time.
        '''

        indx = np.arange(X.shape[0])
        spltr = LeaveOneGroupOut()
        for rep in range(self.n_repeats):

            indx_sample = resample(indx)
            X_sample = X[indx_sample, :]
            y_sample = y[indx_sample]
            g_sample = groups[indx_sample]
            for train, test in spltr.split(X_sample, y_sample, g_sample):
                yield indx_sample[train], indx_sample[test]

class PercentageGroupOut:
    '''
    Custom splitting class which with every iteration of n_repeats it will
    generates a leaveout that has ID and OOD cases based on some grouping (Clusters or chemical grouping).
    For example, if A,B,C 3 chemical groups, a run with percentage = 50% may:
    1. cut [A,B,C] into [A,B] as initial train and [C] initial leaveout chemically
    2. replace 50% of C by data draw from [A,B] WITHOUT REPLACEMENT. [A,B] size shrink
    3. Thus, our new leaveout have data that is in-domain data from [A,B] and OOD from C
    '''

    def __init__(self, n_repeats, groups, percentage, id_tr_ratio, *args, **kwargs):
        '''
        inputs:
            n_repeats =     The number of times to apply splitting.
            groups =        np.array of group classes for the dataset.
            percentage =    percentage of data in leaveout we ensure it is ood
            id_tr_ratio =   the upper bound in_domain test and train ratio we want to ensure
                            (0.2 means in domain test:train= 1:5 ), we want to set this
                            number low so that size(train)>>size(in domain test)
        '''

        self.groups = groups
        self.n_repeats = n_repeats
        self.percentage = percentage
        self.id_tr_ratio = id_tr_ratio

    def get_n_splits(self, X=None, y=None, groups=None):
        '''
        A method to return the O(N) number of splits.
        '''

        self.groups = groups
        self.n_splits = self.n_repeats*len(set(groups))

        return self.n_splits

    def split(self, X=None, y=None, groups=None):
        '''
        For every iteration, bootstrap the original dataset, and leave
        every group out as the testing set one time.
        For the leaveout group, percentage of them is OOD, find same amount of
        ID data from train to form a balance testset.
        '''

        indx = np.arange(X.shape[0])
        spltr = LeaveOneGroupOut()
        for rep in range(self.n_repeats):

            indx_sample = resample(indx)
            X_sample = X[indx_sample, :]
            y_sample = y[indx_sample]
            g_sample = groups[indx_sample]
            # create intial train test split using chemical grouping
            for train, test in spltr.split(X_sample, y_sample, g_sample):
                N_train = train.shape[0]

                # final index to return
                tr_index = None
                test_index = None

                # total number of leaveout (initial cut based on chemical grouping)
                N_test = test.shape[0]

                # number of OOD (based on grouping) we want to keep in initial test
                n_ood = int( N_test * self.percentage )
                # number of in-domain we want to draw from initial train
                n_id = N_test - n_ood

                assert n_id > 0 and n_ood > 0 and N_test == n_ood + n_id
                # we draw n_id # of samples from train WITHOUT REPLACEMENT to form ID test data

                if n_id / ( N_train - n_id) > self.id_tr_ratio:
                    # we ensure that the ratio of id and train after removing id is <= id_tr_ratio
                    # n_id_new = self.id_tr_ratio * (N_train - n_id_new)
                    n_id_new = (self.id_tr_ratio * N_train) / (1 + self.id_tr_ratio)

                    # based on the new in domain size, we update the ood size accoordingly
                    n_ood_new = (self.percentage * n_id_new) / (1 - self.percentage)

                    n_id = int(n_id_new)
                    n_ood = int(n_ood_new)
                    if n_ood > len(test):
                        n_ood = len(test)

                tr, id_test = train_test_split(
                    train, test_size= n_id , random_state=42)

                # we draw n_ood # of samples from test to form OOD test dataset
                _, ood_test = train_test_split(
                    test, test_size= n_ood , random_state=42)

                tr_index = tr.tolist()
                # our final leaveout have both IN/OUT domain data based on chemical grouping
                test_index = id_test.tolist() + ood_test.tolist()

                yield indx_sample[tr_index], indx_sample[test_index]


class PercentageGroupOut2:
    '''
    It enforce the id train ratio
    Custom splitting class which with every iteration of n_repeats it will
    generates a leaveout that has ID and OOD cases based on some grouping (Clusters or chemical grouping).
    For example, if A,B,C 3 chemical groups, a run with percentage = 50% may:
    1. cut [A,B,C] into [A,B] as initial train and [C] initial leaveout chemically
    2. replace 50% of C by data draw from [A,B] WITHOUT REPLACEMENT. [A,B] size shrink
    3. Thus, our new leaveout have data that is in-domain data from [A,B] and OOD from C
    '''

    def __init__(self, n_repeats, groups, percentage, id_tr_ratio, *args, **kwargs):
        '''
        inputs:
            n_repeats =     The number of times to apply splitting.
            groups =        np.array of group classes for the dataset.
            percentage =    percentage of data in leaveout we ensure it is ood
            id_tr_ratio =   the upper bound in_domain test and train ratio we want to ensure
                            (0.2 means in domain test:train= 1:5 ), we want to set this
                            number low so that size(train)>>size(in domain test)
        '''

        self.groups = groups
        self.n_repeats = n_repeats
        self.percentage = percentage
        self.id_tr_ratio = id_tr_ratio

    def get_n_splits(self, X=None, y=None, groups=None):
        '''
        A method to return the O(N) number of splits.
        '''

        self.groups = groups
        self.n_splits = self.n_repeats*len(set(groups))

        return self.n_splits

    def split(self, X=None, y=None, groups=None):
        '''
        For every iteration, bootstrap the original dataset, and leave
        every group out as the testing set one time.
        For the leaveout group, percentage of them is OOD, find same amount of
        ID data from train to form a balance testset.
        '''

        indx = np.arange(X.shape[0])
        spltr = LeaveOneGroupOut()
        for rep in range(self.n_repeats):

            indx_sample = resample(indx)
            X_sample = X[indx_sample, :]
            y_sample = y[indx_sample]
            g_sample = groups[indx_sample]
            # create intial train test split using chemical grouping
            for train, test in spltr.split(X_sample, y_sample, g_sample):
                N_train = train.shape[0]

                # final index to return
                tr_index = None
                test_index = None

                # total number of leaveout (initial cut based on chemical grouping)
                N_test = test.shape[0]

                # number of OOD (based on grouping) we want to keep in initial test
                n_ood = int( N_test * self.percentage )
                # number of in-domain we want to draw from initial train
                n_id = N_test - n_ood

                assert n_id > 0 and n_ood > 0 and N_test == n_ood + n_id
                # we draw n_id # of samples from train WITHOUT REPLACEMENT to form ID test data


                # we ensure that the ratio of id and train after removing id is == id_tr_ratio
                # n_id_new = self.id_tr_ratio * (N_train - n_id_new)

                n_id_new = (self.id_tr_ratio * N_train) / (1 + self.id_tr_ratio)

                # based on the new in domain size, we update the ood size accoordingly
                n_ood_new = (self.percentage * n_id_new) / (1 - self.percentage)

                n_id = int(n_id_new)
                n_ood = int(n_ood_new)
                if n_ood > len(test):
                    n_ood = len(test)


                tr, id_test = train_test_split(
                    train, test_size= n_id , random_state=42)

                # we draw n_ood # of samples from test to form OOD test dataset
                if n_ood != len(test):
                    _, ood_test = train_test_split(
                        test, test_size= n_ood , random_state=42)
                else:
                    # using all testdata
                    ood_test = test

                tr_index = tr.tolist()
                # our final leaveout have both IN/OUT domain data based on chemical grouping
                test_index = id_test.tolist() + ood_test.tolist()

                yield indx_sample[tr_index], indx_sample[test_index]

class ClusterSplit:
    '''
    Custom splitting class which pre-clusters data and then split.
    '''

    def __init__(self, clust, *args, **kwargs):
        '''
        inputs:
            clust = The class of cluster from Scikit-learn.
        '''

        self.clust = clust(*args, **kwargs)

        # Make sure it runs in serial
        if hasattr(self.clust, 'n_jobs'):
            self.clust.n_jobs = 1

    def get_n_splits(self, X=None, y=None, groups=None):
        '''
        A method to return the number of splits.
        '''

        return self.n_splits

    def split(self, X, y=None, groups=None):
        '''
        Cluster data, randomize cluster order, randomize case order,
        and then split into train and test sets self.reps number of times.

        inputs:
            X = The features.
        outputs:
            A generator for train and test splits.
        '''

        self.clust.fit(X)  # Do clustering

        # Get splits based on cluster labels
        df = pd.DataFrame(X)
        df['cluster'] = self.clust.labels_
        cluster_order = list(set(self.clust.labels_))
        self.n_splits = len(cluster_order)

        # Shuffle data
        random.shuffle(cluster_order)
        df = df.sample(frac=1)

        # Randomize cluster order
        df = [df.loc[df['cluster'] == i] for i in cluster_order]

        # Do for requested repeats
        for i in range(self.n_splits):

            te = df[i]  # Test
            tr = pd.concat(df[:i]+df[i+1:])  # Train

            # Get the indexes
            train = tr.index.tolist()
            test = te.index.tolist()

            yield train, test


class RepeatedClusterSplit:
    '''
    Custom splitting class which pre-clusters data and then splits
    to folds.
    '''

    def __init__(self, clust, n_repeats, *args, **kwargs):
        '''
        inputs:
            clust = The class of cluster from Scikit-learn.
            n_repeats = The number of times to apply splitting.
        '''

        self.clust = clust(*args, **kwargs)

        # Make sure it runs in serial
        if hasattr(self.clust, 'n_jobs'):
            self.clust.n_jobs = 1

        self.n_repeats = n_repeats

    def get_n_splits(self, X=None, y=None, groups=None):
        '''
        A method to return the number of splits.
        '''

        return self.n_splits*self.n_repeats

    def split(self, X, y=None, groups=None):
        '''
        Cluster data, randomize cluster order, randomize case order,
        and then split into train and test sets self.reps number of times.

        inputs:
            X = The features.
        outputs:
            A generator for train and test splits.
        '''

        self.clust.fit(X)  # Do clustering

        # Get splits based on cluster labels
        df = pd.DataFrame(X)
        df['cluster'] = self.clust.labels_
        cluster_order = list(set(self.clust.labels_))
        random.shuffle(cluster_order)
        df = df.sample(frac=1)

        # Randomize cluster order
        df = [df.loc[df['cluster'] == i] for i in cluster_order]

        self.n_splits = len(cluster_order)
        range_splits = range(self.n_splits)

        # Do for requested repeats
        for rep in range(self.n_repeats):

            sub = [i.sample(frac=1) for i in df]  # Shuffle
            for i in range_splits:

                te = sub[i]  # Test
                tr = pd.concat(sub[:i]+sub[i+1:])  # Train

                # Get the indexes
                train = tr.index.tolist()
                test = te.index.tolist()

                yield train, test


class RepeatedPDFSplit:
    '''
    Custom splitting class which groups data on a multivariate probability
    distribution function and then splits to folds. Folds should have
    least probable cases.
    '''

    def __init__(self, frac, n_repeats, *args, **kwargs):
        '''
        inputs:
            frac = The fraction of the split.
            n_repeats = The number of times to apply splitting.
        '''

        self.frac = frac
        self.n_repeats = n_repeats

    def get_n_splits(self, X=None, y=None, groups=None):
        '''
        A method to return the number of splits.
        '''

        return self.n_repeats

    def split(self, X, y=None, groups=None):
        '''
        Cluster data, randomize cluster order, randomize case order,
        and then split into train and test sets self.reps number of times.

        inputs:
            X = The features.
        outputs:
            A generator for train and test splits.
        '''

        for i in range(self.n_repeats):

            # Sample with replacement
            X_sample = resample(X)

            # Estimate bandwidth
            grid = {
                    'kernel': [
                               'gaussian',
                               'tophat',
                               'epanechnikov',
                               'exponential',
                               'linear',
                               'cosine'
                               ],
                    'bandwidth': [estimate_bandwidth(X_sample)]
                    }
            model = GridSearchCV(
                                 KernelDensity(),
                                 grid,
                                 cv=5,
                                 )

            model.fit(X_sample)

            dist = model.score_samples(X_sample)  # Natural log distance

            df = {'dist': dist, 'index': list(range(X_sample.shape[0]))}
            df = pd.DataFrame(df)
            df.sort_values(by='dist', inplace=True)

            split = int(df.shape[0]*self.frac)
            test = df[split:].index.tolist()
            train = df[:split].index.tolist()

            yield train, test
