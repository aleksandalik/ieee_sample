%matplotlib inline
import os
import random
import itertools
from tqdm import tqdm
import gc
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold, KFold, TimeSeriesSplit, train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import seaborn as sns
import lightgbm as lgb
from datetime import datetime, timedelta

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
path = r'C:\Users\User\AppData\Local\Programs\Python\Python37\Scripts\kaggle\\'


#Загружаем данные
dtypes_id = pd.read_csv(path+'dtypes_id.csv',header=None)
dtypes_trans = pd.read_csv(path+'dtypes_trans.csv',header=None)
dtypes_id = dict(zip(dtypes_id[0],dtypes_id[1]))
dtypes_trans = dict(zip(dtypes_trans[0],dtypes_trans[1]))

train_id = pd.read_csv(path+'train_identity.csv', dtype=dtypes_id)
train_trans = pd.read_csv(path+'train_transaction.csv', dtype=dtypes_trans)
train = train_trans.merge(train_id, how='left',on='TransactionID')
y_train = train['isFraud']
del train_trans

test_id = pd.read_csv(path+'test_identity.csv', dtype=dtypes_id)
test_trans = pd.read_csv(path+'test_transaction.csv', dtype=dtypes_trans)
test = test_trans.merge(test_id, how='left',on='TransactionID')
del test_trans
dtypes_id = pd.read_csv(path+'dtypes_id.csv',header=None)
dtypes_trans = pd.read_csv(path+'dtypes_trans.csv',header=None)
dtypes_id = dict(zip(dtypes_id[0],dtypes_id[1]))
dtypes_trans = dict(zip(dtypes_trans[0],dtypes_trans[1]))
​
train_id = pd.read_csv(path+'train_identity.csv', dtype=dtypes_id)
train_trans = pd.read_csv(path+'train_transaction.csv', dtype=dtypes_trans)
train = train_trans.merge(train_id, how='left',on='TransactionID')
y_train = train['isFraud']
del train_trans
​
test_id = pd.read_csv(path+'test_identity.csv', dtype=dtypes_id)
test_trans = pd.read_csv(path+'test_transaction.csv', dtype=dtypes_trans)
test = test_trans.merge(test_id, how='left',on='TransactionID')
del test_trans

###########################################################################################
#работа с датами; выделение частей datetime для составления новых признаков (какие-нибудь могут оказаться полезны)

def hour_rounded(dtime):
    return (dtime.replace(second=0, microsecond=0, minute=0, hour=dtime.hour)
               +timedelta(hours=dtime.minute//30)).hour
#chrome 71.0 for ios - 28 ноября и пики выходят на праздники
startdate = datetime.strptime('2017-12-01 00:00:00', '%Y-%m-%d %H:%M:%S')

train.insert(2,'Date',train['TransactionDT'].apply(lambda x: (startdate + timedelta(seconds = x))))
train.insert(2,'DayOfMonth',train.Date.apply(lambda x: x.day))
train.insert(2,'DayOfWeek',train.Date.apply(lambda x: x.weekday()))
train.insert(2,'DayOfYear',train.Date.apply(lambda x: x.timetuple().tm_yday))
train.insert(2,'Minute',train.Date.apply(lambda x: x.minute))
train.insert(2,'Hour',train.Date.apply(lambda x: x.hour))
train.insert(2,'Hour_rounded',train.Date.apply(hour_rounded))
train.insert(2,'Morning',np.where(train['Hour'].between(3,11), 1, 0))
train.insert(2,'Evening',np.where(train['Hour'].between(12,16), 1, 0))
train.insert(2,'Late_Evening',np.where((train['Hour'].between(17,23)
                                       | (train['Hour'].between(0,2))), 1, 0))
train.insert(2,'Date_clear', pd.to_datetime(train.Date.apply(lambda x: x.date())))

#mean isFraud on every day of Year (created because test and train datasets have december in common)
train['FraudToDate'] = train.merge(train[['DayOfYear','isFraud']].groupby('DayOfYear').agg(['count','mean']).droplevel(0,1).sort_values('count').reset_index(0), 
                                   how='left',
                                   on='DayOfYear')['mean']

###########################################################################################
#визуализация покажет, как лучше выделить моменты дня в качестве разработки новых признаков
fig, ax = plt.subplots()
ax2 = ax.twinx()
sns.distplot(train.Hour,bins=24, kde=False, ax=ax)
l = sns.lineplot(train[['Hour','isFraud']].groupby('Hour').agg(['mean']).index,
                train[['Hour','isFraud']].groupby('Hour').agg(['mean']).values.reshape(1,-1)[0], ax=ax2)
l.set(xticks=np.arange(0,25));


###########################################################################################
# Колонки с большим количеством NaN (90% пропусков устанавливаем наверняка, чтобы не выкинуть лишнего)
train_cols_na = [col for col in train.columns if train.iloc[:sep][col].isnull().sum() 
                  / train.iloc[:sep].shape[0] > 0.9]
test_cols_na = [col for col in train.columns if train.iloc[sep:][col].isnull().sum() 
                       / train.iloc[sep:].shape[0] > 0.9]

cols_to_drop = list(set(test_cols_na+train_cols_na)-set(['isFraud']))

train.drop(cols_to_drop, axis=1, inplace=True)
#test.drop(cols_to_drop, axis=1, inplace=True)


###########################################################################################
# Предварительно кластеризуем транзакции для нахождения отдельных юзеров (используя данные по карте; по адресу и т.д)
# Ниже приводится составление признаков по каждому юзеру/покупателю

uid_type = 'uid'

train['TransactionAmt_prev_uid'] = train.groupby(uid_type)['TransactionAmt'].shift()
train['TransactionAmt_mean_uid'] = pd.Series(np.array([train.groupby(uid_type)['TransactionAmt'].shift(),
                                                       train.groupby(uid_type)['TransactionAmt'].shift(2),
                                                       train.groupby(uid_type)['TransactionAmt'].shift(3)]).T.mean(1))

train['TransactionAmt_diff'] = train['TransactionAmt'] - train['TransactionAmt_prev_uid']
train['TransactionAmt_mean_diff'] = train['TransactionAmt'] - train['TransactionAmt_mean_uid']

train['AmtDecimal_kind_prev_uid'] = pd.Series(np.array([train.groupby(uid_type)['AmtDecimal_kind'].shift(),
                                                          train.groupby(uid_type)['AmtDecimal_kind'].shift(2),
                                                          train.groupby(uid_type)['AmtDecimal_kind'].shift(3)]).T.mean(1))

train['AmtDecimal_kind_same'] = np.select([(train['AmtDecimal_kind_prev_uid']!=0) &
                                              (train['AmtDecimal_kind']!=0),
                                              (train['AmtDecimal_kind_prev_uid']==0) &
                                              (train['AmtDecimal_kind']!=0),
                                              (train['AmtDecimal_kind_prev_uid']!=0) &
                                              (train['AmtDecimal_kind']==0),
                                              (train['AmtDecimal_kind_prev_uid']==0) &
                                              (train['AmtDecimal_kind']==0)],[0,1,2,3], default = np.nan)

train['AmtDecimal_kind_diff'] = train['AmtDecimal_kind'] - train['AmtDecimal_kind_prev_uid']
train['ProductCD_prev_uid'] = train.groupby(uid_type)['ProductCD'].shift()

train['dist1_prev_uid'] = train.groupby(uid_type)['dist1'].shift()

train['R_emaildomain_prev_uid'] = train.groupby(uid_type)['R_emaildomain'].shift()

train['Date_prev_uid'] = train.groupby(uid_type)['Date'].shift()

train['Date_prev_uid_diff_hours'] = (train['Date'] - train['Date_prev_uid']).apply(lambda x: x.seconds/3600)


###########################################################################################
#В данном блоке проводилось составление новых признаков, а также агрегирование знаений по группам (mean, count, sum);
#Также в этом блоке проводилась нормализация численных признаков.
###########################################################################################

###########################################################################################
#В блоке ниже проводилось обучение модели (кросс-валидация) при имеющихся параметрах (из публичных kernel на kaggle) на данный момент.
params = {'num_leaves': 491,
          'min_child_weight': 0.03454472573214212,
          'feature_fraction': 0.3797454081646243,
          'bagging_fraction': 0.4181193142567742,
          'min_data_in_leaf': 106,
          'objective': 'binary',
          'max_depth': -1,
          'learning_rate': 0.006883242363721497,
          "boosting_type": "gbdt",
          "bagging_seed": 11,
          "metric": 'auc',
          "verbosity": -1,
          'reg_alpha': 0.3899927210061127,
          'reg_lambda': 0.6485237330340494,
          'random_state': 47
         }

splits = 4
folds = KFold(n_splits = splits)
oof = np.zeros(len(train[c]))
predictions = np.zeros(len(target))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train[c].values, target.values)):
    gc.collect()
    print("Fold {}".format(fold_))
    train_df, y_train_df = train[c].iloc[trn_idx],target.iloc[trn_idx]
    valid_df, y_valid_df = train[c].iloc[val_idx], target.iloc[val_idx]
    
    trn_data = lgb.Dataset(train_df, label=y_train_df)
    val_data = lgb.Dataset(valid_df, label=y_valid_df)
    
    clf = lgb.train(params,
                    trn_data,
                    7000,
                    valid_sets = [trn_data, val_data],
                    verbose_eval=300,
                    early_stopping_rounds=300)

    pred = clf.predict(valid_df)
    oof[val_idx] = pred
    print( "  auc = ", roc_auc_score(y_valid_df, pred) )
    predictions += clf.predict(test.drop('TransactionID',1)) / splits

###########################################################################################
Запись предсказаний по тестовым данным
sample_submission = pd.Series(predictions)
sample_submission.to_csv("lgb_sub.csv", index=False)