import numpy as np
import os
import xgboost as xgb

def are(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred) / y_true)

def msre(y_true, y_pred):
    return np.mean(((y_true - y_pred) / y_true) ** 2)

def rmrse(y_true, y_pred):
    return np.sqrt(msre(y_true, y_pred))

def eval(y_true, y_pred):
    return are(y_true, y_pred), rmrse(y_true, y_pred)

if __name__ == '__main__':

    pt = 's'
    mapping = {'s': 0, 'm': 1}

    feats = np.load('glb_feats.npy')
    targets = np.load('all_targets.npy')[:, mapping[pt]]

    index = np.load('newidx.npy')
    large_idx = np.load('lrg_test_idx.npy')

    num_test = int(0.1 * len(index))

    results = []
    for cf in range(10):
        test_idx = index[cf*num_test : (cf+1)*num_test]
        other_idx = np.append(index[(cf+1)*num_test:], index[:num_test])

        valid_idx = other_idx[:num_test]
        train_idx = other_idx[num_test:]

        valid_set = (feats[valid_idx], targets[valid_idx])
        train_set = (feats[train_idx], targets[train_idx])

        foldresult = []

        for exp in range(5):

            regressor=xgb.XGBRegressor(learning_rate = 0.015,
                                       n_estimators  = 700,
                                       max_depth     = 5)

            regressor.fit(feats[train_idx], targets[train_idx], eval_metric='rmsle',
                          eval_set=[train_set, valid_set], early_stopping_rounds=20, verbose=False)

            test_pred = regressor.predict(feats[test_idx])

            tare, trmrse = eval(targets[test_idx], test_pred)

            test_large = regressor.predict(feats[large_idx])

            lare, lrmrse = eval(targets[large_idx], test_large)

            foldresult.append([tare, trmrse, lare, lrmrse])

        tare, trmrse, lare, lrmrse = np.mean(foldresult, 0)

        print('Fold {:d} | Test ARE {:.2f} | Test RMRSE {:.2f} | Large ARE {:.2f}| Large RMRSE {:.2f}'.format(cf, tare*100, trmrse*100, lare*100, lrmrse*100))

        results.append(foldresult)

    results = np.array(results)

    print(np.mean(results.mean(0)), 0)
    print(np.var(results.reshape(-1, 4)), 0)

    np.save(pt+'result/xgb_valid.npy', results)

