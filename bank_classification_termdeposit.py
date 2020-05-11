# -*- coding: utf-8 -*-
'''
分類模型: 預測是否會開定期存款戶頭(透過行銷活動)

特徵(x):
1. bank client data:
    1 - age: (numeric)
    2 - job: type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
    3 - marital: marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
    4 - education: (categorical: primary, secondary, tertiary and unknown)
    5 - default: has credit in default? (categorical: 'no','yes','unknown')
    6 - housing: has housing loan? (categorical: 'no','yes','unknown')
    7 - loan: has personal loan? (categorical: 'no','yes','unknown')
    8 - balance: Balance of the individual.

2. Related with the last contact of the current campaign:
    9 - contact: contact communication type (categorical: 'cellular','telephone')
    10 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
    ?11 - day: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
    day: 日期，非星期幾(numeric，1~31)
    12 - duration通話時間: last contact duration, in seconds (numeric). 
    Important note: this attribute highly affects the output target 
    (e.g., if duration=0 then y='no'). Yet, the duration is not known 
    before a call is performed. Also, after the end of the call y is 
    obviously known. Thus, this input should only be included for benchmark 
    purposes and should be discarded if the intention is to have a realistic 
    predictive model.

3. other attributes:
    13 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
    14 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
    15 - previous: number of contacts performed before this campaign and for this client (numeric)
    16 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')

y: has the client subscribed a term deposit? (binary: 'yes','no')

'''
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv('bank.csv') # raw_data
data_copy=data.copy()

data.head()
data.info() # check if there are missing values=> no
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 11162 entries, 0 to 11161
Data columns (total 16 columns):
age          11162 non-null int64
job          11162 non-null object
marital      11162 non-null object
education    11162 non-null object
default      11162 non-null object
balance      11162 non-null int64
housing      11162 non-null object
loan         11162 non-null object
contact      11162 non-null object
day          11162 non-null int64
month        11162 non-null object
campaign     11162 non-null int64
pdays        11162 non-null int64
previous     11162 non-null int64
poutcome     11162 non-null object
deposit      11162 non-null object
dtypes: int64(6), object(10)
memory usage: 1.4+ MB
'''

#%%
'''
========================================
資料探索-數值型資料
========================================

'''

data_describe=data.describe()
'''
數值(浮點數)皆做無條件捨去處理
1. 樣本數: 11162
2. 年齡=>青壯年+中年(30~63歲, 81.5%)
平均為41歲，中位數為39歲，標準差為11；
最小值為18，最大值為95；
約68%樣本介於30~52歲，約95%樣本介於18~63歲。
note: 較年輕的人是否較容易辦理定存? (應比較年輕人辦理各種金融商品的比例)
3. 餘額=> 高標準差，分布極廣
平均為1528，中位數為550，標準差為3225；
最小值為-6847，最大值為81204；
約68%樣本介於-1697~4753，約95%樣本介於-4922~7978。
note: 負數餘額的客戶是否較不容易辦理定存?
4. 行銷活動特徵=>需進行進一步處理(視覺化)
previous vs. poutcome: (剔除previous=0)
前次活動聯絡次數對前次活動成功率影響?
campaign vs. deposit:
此次活動聯絡次數對此次活動成功率影響?
previous + campaign vs. deposit: (剔除previous=0)
前次+此次活動聯絡次數對此次活動成功率影響?
pdays + previous + campaign vs. deposit: (剔除previous=0)
前次活動最後連絡距今天數+前次活動聯絡次數+此次活動聯絡次數
對此次成功率影響?
'''

#data=data.drop(columns=['duration']) 
# remove duration, why? 1. high duration increases the probability of opening term deposit
# 2. only obtained after the phone call is performed

'''
========================================
資料探索-類別型資料
========================================
'''

'''
相關性
'''
from sklearn.preprocessing import LabelEncoder
data['deposit_trans'] = LabelEncoder().fit_transform(data['deposit'])
data['job_trans'] = LabelEncoder().fit_transform(data['job'])
data['marital_trans'] = LabelEncoder().fit_transform(data['marital'])
data['education_trans'] = LabelEncoder().fit_transform(data['education'])
data['default_trans'] = LabelEncoder().fit_transform(data['default'])
data['housing_trans'] = LabelEncoder().fit_transform(data['housing'])
data['loan_trans'] = LabelEncoder().fit_transform(data['loan'])
#data['contact_trans'] = LabelEncoder().fit_transform(data['contact'])
#data['month_trans'] = LabelEncoder().fit_transform(data['month'])

trans_df = data.select_dtypes(exclude="object") # 取出numeric類型的特徵
#trans_df = data.select_dtypes(include="object") # 取出包含類別類型的特徵

data_corr=trans_df.corr().astype(float)
 
# plot pairwise data relationship
pair_corr= sns.PairGrid(data) 
pair_corr.map(plt.scatter)
plt.title('Pairwise Data Relationship')
plt.savefig('pair_corr1.png')
# =>特徵間沒明顯有相關性

# plot heatmap of correlation metrix
plt.figure(figsize=(20,8))
sns.heatmap(data_corr, cmap='Blues', annot=True, annot_kws={"size": 10},
            xticklabels=data_corr.columns.values,
            yticklabels=data_corr.columns.values)
plt.title('Correlation Matrix')            
plt.savefig('heatmap_corr3.png')   
# =>特徵間沒有相關性(相關性係數>0.7)
# => 通話時間與是否開定存的相關性為45%；
#上次活動聯絡次數與上一通電話至今的天數的相關性為51%
# => 是否有貸款和是否有房產分別與是否開定存有-11%和-20%的相關性

#%%
'''
特徵間的關係
'''
sns.lmplot(x='age', y='balance', data=data, row='default', col='loan', hue='deposit')
#plt.title('balance vs. age by deposit and loan')
plt.savefig('balancevs.age_bydepositandloananddefault.png')
# =>年齡與餘額沒有太大的相關性(11%)
# =>有貸款的或有信用違約的人餘額較低，違約的人餘額普遍較有貸款的人低

# create new column 'balance status'
high_level=data['balance'].describe()['75%']
low_level=data['balance'].describe()['25%']
data['bal_status']=data['balance'].apply(lambda x: 'high' if x >=high_level 
    else ('low' if x< low_level else 'middle' ))

# 高/低餘額人數比例
#https://github.com/mwaskom/seaborn/issues/1027
def highlow_balance_perc(data, col_name):
    names=data[col_name].unique()
    sector=data.groupby(col_name)
    highlow_perc=[]
    for i in range(len(names)):
        tem=sector.get_group(names[i])    
        high_perc=tem['bal_status'].value_counts()['high']/len(tem)
        low_perc=tem['bal_status'].value_counts()['low']/len(tem)
        ans=names[i], high_perc, low_perc, high_perc-low_perc
        highlow_perc.append(ans)
    highlow_perc_df=pd.DataFrame(highlow_perc, index=names, columns=[col_name, 'high', 'low', 'diff(h-l)'])
    return highlow_perc_df

def saveplot_highlow_balstatus(data, col_name, figsize=(10,5)):
    high_low_perc=highlow_balance_perc(data, col_name)
    high_low_perc.plot.barh(figsize=figsize)
    plt.title('The percentage of low/high balance level in each %s status' % col_name)
    plt.ylabel(col_name)
    #plt.savefig('%s_perc_bybalstatus.png' % col_name, dpi=300)

saveplot_highlow_balstatus(data, 'education')
# => 大學畢業以上的人低餘額人數比例較高餘額人數比例低
saveplot_highlow_balstatus(data, 'loan')
# => 有貸款的人低餘額人數比例較高餘額人數比例高
saveplot_highlow_balstatus(data, 'marital')
# => 離婚的人低餘額人數比例較高餘額人數比例高
saveplot_highlow_balstatus(data, 'job')
# => 退休人士、管理階層、學生與自僱者的高餘額人數比例與低於額人數比例差異較多
saveplot_highlow_balstatus(data, 'default')
# => 違約的人低餘額人數比例較高餘額人數比例高
saveplot_highlow_balstatus(data, 'housing')
# => 有房產的人低餘額人數比例較高餘額人數比例高

# create new column "duration_level"
data['duration_level']=data['duration'].apply(lambda x: 'above' 
    if x >= data['duration'].mean() else 'below')

ax=sns.countplot(y='deposit', hue='duration_level', data=data)
#show percentage on countplot
c=1
for p in ax.patches:
    if c <= 2:
        total=len(data[data['deposit'] == 'yes'])
    else:
        total=len(data[data['deposit'] == 'no'])        
    percentage = '{:.1f}%'.format(100 * p.get_width()/total)
    x = p.get_x() + p.get_width() + 0.02    
    y = p.get_y() + p.get_height()/2
    ax.annotate(percentage, (x, y))
    c += 1
plt.title('The distribubtion of deposit by duration status')
plt.savefig('deposit_count_bydurationstatus.png', dpi=300)
# => 在開定存的人中，有56.2%的人通話時間高於平均；
# 在沒開定存的人中，有85.2%的人通話時間低於平均

def highlow_deposit_perc(data, col_name):
    names=data[col_name].unique()
    sector=data.groupby(col_name)
    highlow_perc=[]
    for i in range(len(names)):
        tem=sector.get_group(names[i])    
        high_perc=tem['deposit'].value_counts()['yes']/len(tem)
        low_perc=tem['deposit'].value_counts()['no']/len(tem)
        ans=names[i], high_perc, low_perc, high_perc-low_perc
        highlow_perc.append(ans)
    highlow_perc_df=pd.DataFrame(highlow_perc, index=names, columns=[col_name, 'yes', 'no', 'diff(y-n)'])
    return highlow_perc_df

def saveplot_highlow_deposit(data, col_name, figsize=(10,5)):
    high_low_perc=highlow_deposit_perc(data, col_name)
    high_low_perc.plot.barh(figsize=figsize)
    plt.title('The percentage of yes/no deposit in each %s status' % col_name)
    plt.ylabel(col_name)
    plt.savefig('%s_perc_bydeposit.png' % col_name, dpi=300)

saveplot_highlow_deposit(data, 'job', figsize=(25,10))

def balance_deposit_perc(data, col_name):
    names=data[col_name].unique()
    sector=data.groupby(col_name)
    highlow_perc=[]
    for i in range(len(names)):
        tem=sector.get_group(names[i])    
        high_perc=tem['deposit'].value_counts()['yes']/len(tem)
        low_perc=tem['deposit'].value_counts()['no']/len(tem)
        ans=names[i], high_perc, low_perc, high_perc-low_perc
        highlow_perc.append(ans)
    highlow_perc_df=pd.DataFrame(highlow_perc, index=names, columns=[col_name, 'yes', 'no', 'diff(y-n)'])
    return highlow_perc_df

def saveplot_highlow_deposit(data, col_name, figsize=(10,5)):
    high_low_perc=balance_deposit_perc(data, col_name)
    high_low_perc.plot.barh(figsize=figsize)
    plt.title('The percentage of yes/no deposit in each balance status')
    plt.ylabel(col_name)
    plt.savefig('%s_perc_bydeposit.png' % col_name, dpi=300)

saveplot_highlow_deposit(data, 'bal_status', figsize=(18,10))

'''
data1=data[(data['default'] == 'no')& (data['loan'] == 'no') ]
data1['deposit'].value_counts() # no:4823 yes:4768
sns.countplot(x='bal_status', data=data1)
# =>在沒有貸款也沒有違約的人中，開定存的人餘額較不開定存的人高

plt.figure(figsize=(20,10)) # (長, 寬)
sns.countplot(y='job', data=data, hue='deposit')
plt.title('The number of occupation by deposit')
plt.savefig('job_count_bydeposit.png', dpi=300)
# =>職業為管理階層的人佔比最高

plt.figure(figsize=(15,8))
sns.violinplot(x='job', y='balance', data=data, hue='deposit')
plt.title('balance vs. job by deposit')
plt.savefig('balance_job_bydeposit.png', dpi=300)
# => 退休人士存款餘額普遍較高(分布較高)

plt.figure(figsize=(15,8))
sns.countplot(y='job', data=data, hue='bal_status')
plt.title('The number of occupation by balance status')
plt.savefig('job_count_bybalstatus.png', dpi=300)
# => 管理階層和退休人士的存款餘額普遍較高

#sns.violinplot(x='education', y='balance', data=data, hue='deposit')
# => 教育程度在餘額分布上沒有顯著差異
'''

#%%
'''
========================================
分類模型
========================================
'''
dep=data['deposit']

# 分層抽樣 (Stratified Sampling) 交叉驗證 StratifiedShuffleSplit
#https://www.twblogs.net/a/5c25c8e2bd9eee16b3db7e75
from sklearn.model_selection import StratifiedShuffleSplit
# Here we split the data into training and test sets and implement a stratified shuffle split.
stratified = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# => 是否有貸款和是否有房產分別與是否開定存有-11%和-20%的相關性
data['loan'].value_counts()/len(data)
#no     0.869199 yes    0.130801 
#=>是與否比例差距大=>在分割資料集時，為了避免訓練或測試集中，都是yes or no的情況，
# 以loan為基準做交叉驗證分割資料集
data['housing'].value_counts()/len(data)
#no     0.526877 yes    0.473123

for train_set, test_set in stratified.split(data_copy, data_copy["loan"]):
    stratified_train = data_copy.loc[train_set]
    stratified_test = data_copy.loc[test_set]
    
stratified_train["loan"].value_counts()/len(data_copy)
#no     0.695306 yes    0.104641 (6.644680383406122)
stratified_test["loan"].value_counts()/len(data_copy)
#no     0.173894 yes    0.026160 (6.647324159021407)
# =>使用分層+隨機 交叉驗證，訓練集與測試集中，loan結果的比例一樣

train_data=stratified_train
test_data=stratified_test

#%%
from sklearn.base import BaseEstimator, TransformerMixin
#https://medium.com/@weilihmen/%E9%97%9C%E6%96%BCpython%E7%9A%84%E9%A1%9E%E5%88%A5-class-%E5%9F%BA%E6%9C%AC%E7%AF%87-5468812c58f2
#https://codertw.com/%E4%BA%BA%E5%B7%A5%E6%99%BA%E6%85%A7/7764/

# 因sklearn不能直接處理DataFrames，自定義一個處理的方法將其轉化為numpy類型
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names]

#%%
'''
自行貼上CategoricalEncoder class，因版本問題
RuntimeError: CategoricalEncoder briefly existed in 0.20dev. Its functionality has been rolled into the OneHotEncoder and OrdinalEncoder. This stub will be removed in version 0.21.
'''
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.preprocessing import LabelEncoder
from scipy import sparse

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical features as a numeric array.
    The input to this transformer should be a matrix of integers or strings,
    denoting the values taken on by categorical (discrete) features.
    The features can be encoded using a one-hot aka one-of-K scheme
    (``encoding='onehot'``, the default) or converted to ordinal integers
    (``encoding='ordinal'``).
    This encoding is needed for feeding categorical data to many scikit-learn
    estimators, notably linear models and SVMs with the standard kernels.
    Read more in the :ref:`User Guide <preprocessing_categorical_features>`.
    Parameters
    ----------
    encoding : str, 'onehot', 'onehot-dense' or 'ordinal'
        The type of encoding to use (default is 'onehot'):
        - 'onehot': encode the features using a one-hot aka one-of-K scheme
          (or also called 'dummy' encoding). This creates a binary column for
          each category and returns a sparse matrix.
        - 'onehot-dense': the same as 'onehot' but returns a dense array
          instead of a sparse matrix.
        - 'ordinal': encode the features as ordinal integers. This results in
          a single column of integers (0 to n_categories - 1) per feature.
    categories : 'auto' or a list of lists/arrays of values.
        Categories (unique values) per feature:
        - 'auto' : Determine categories automatically from the training data.
        - list : ``categories[i]`` holds the categories expected in the ith
          column. The passed categories are sorted before encoding the data
          (used categories can be found in the ``categories_`` attribute).
    dtype : number type, default np.float64
        Desired dtype of output.
    handle_unknown : 'error' (default) or 'ignore'
        Whether to raise an error or ignore if a unknown categorical feature is
        present during transform (default is to raise). When this is parameter
        is set to 'ignore' and an unknown category is encountered during
        transform, the resulting one-hot encoded columns for this feature
        will be all zeros.
        Ignoring unknown categories is not supported for
        ``encoding='ordinal'``.
    Attributes
    ----------
    categories_ : list of arrays
        The categories of each feature determined during fitting. When
        categories were specified manually, this holds the sorted categories
        (in order corresponding with output of `transform`).
    Examples
    --------
    Given a dataset with three features and two samples, we let the encoder
    find the maximum value per feature and transform the data to a binary
    one-hot encoding.
    >>> from sklearn.preprocessing import CategoricalEncoder
    >>> enc = CategoricalEncoder(handle_unknown='ignore')
    >>> enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])
    ... # doctest: +ELLIPSIS
    CategoricalEncoder(categories='auto', dtype=<... 'numpy.float64'>,
              encoding='onehot', handle_unknown='ignore')
    >>> enc.transform([[0, 1, 1], [1, 0, 4]]).toarray()
    array([[ 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.],
           [ 0.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.]])
    See also
    --------
    sklearn.preprocessing.OneHotEncoder : performs a one-hot encoding of
      integer ordinal features. The ``OneHotEncoder assumes`` that input
      features take on values in the range ``[0, max(feature)]`` instead of
      using the unique values.
    sklearn.feature_extraction.DictVectorizer : performs a one-hot encoding of
      dictionary items (also handles string-valued features).
    sklearn.feature_extraction.FeatureHasher : performs an approximate one-hot
      encoding of dictionary items or strings.
    """

    def __init__(self, encoding='onehot', categories='auto', dtype=np.float64,
                 handle_unknown='error'):
        self.encoding = encoding
        self.categories = categories
        self.dtype = dtype
        self.handle_unknown = handle_unknown

    def fit(self, X, y=None):
        """Fit the CategoricalEncoder to X.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_feature]
            The data to determine the categories of each feature.
        Returns
        -------
        self
        """
        if self.encoding not in ['onehot', 'onehot-dense', 'ordinal']:
            template = ("encoding should be either 'onehot', 'onehot-dense' "
                        "or 'ordinal', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.handle_unknown not in ['error', 'ignore']:
            template = ("handle_unknown should be either 'error' or "
                        "'ignore', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.encoding == 'ordinal' and self.handle_unknown == 'ignore':
            raise ValueError("handle_unknown='ignore' is not supported for"
                             " encoding='ordinal'")

        X = check_array(X, dtype=np.object, accept_sparse='csc', copy=True)
        n_samples, n_features = X.shape

        self._label_encoders_ = [LabelEncoder() for _ in range(n_features)]

        for i in range(n_features):
            le = self._label_encoders_[i]
            Xi = X[:, i]
            if self.categories == 'auto':
                le.fit(Xi)
            else:
                valid_mask = np.in1d(Xi, self.categories[i])
                if not np.all(valid_mask):
                    if self.handle_unknown == 'error':
                        diff = np.unique(Xi[~valid_mask])
                        msg = ("Found unknown categories {0} in column {1}"
                               " during fit".format(diff, i))
                        raise ValueError(msg)
                le.classes_ = np.array(np.sort(self.categories[i]))

        self.categories_ = [le.classes_ for le in self._label_encoders_]

        return self

    def transform(self, X):
        """Transform X using one-hot encoding.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to encode.
        Returns
        -------
        X_out : sparse matrix or a 2-d array
            Transformed input.
        """
        X = check_array(X, accept_sparse='csc', dtype=np.object, copy=True)
        n_samples, n_features = X.shape
        X_int = np.zeros_like(X, dtype=np.int)
        X_mask = np.ones_like(X, dtype=np.bool)

        for i in range(n_features):
            valid_mask = np.in1d(X[:, i], self.categories_[i])
            if not np.all(valid_mask):
                if self.handle_unknown == 'error':
                    diff = np.unique(X[~valid_mask, i])
                    msg = ("Found unknown categories {0} in column {1}"
                           " during transform".format(diff, i))
                    raise ValueError(msg)
                else:
                    # Set the problematic rows to an acceptable value and
                    # continue `The rows are marked `X_mask` and will be
                    # removed later.
                    X_mask[:, i] = valid_mask
                    X[:, i][~valid_mask] = self.categories_[i][0]
            X_int[:, i] = self._label_encoders_[i].transform(X[:, i])

        if self.encoding == 'ordinal':
            return X_int.astype(self.dtype, copy=False)

        mask = X_mask.ravel()
        n_values = [cats.shape[0] for cats in self.categories_]
        n_values = np.array([0] + n_values)
        indices = np.cumsum(n_values)

        column_indices = (X_int + indices[:-1]).ravel()[mask]
        row_indices = np.repeat(np.arange(n_samples, dtype=np.int32),
                                n_features)[mask]
        data = np.ones(n_samples * n_features)[mask]
        out = sparse.csc_matrix((data, (row_indices, column_indices)),
                                shape=(n_samples, indices[-1]),
                                dtype=self.dtype).tocsr()
        if self.encoding == 'onehot-dense':
            return out.toarray()
        else:
            return out

#%%
# pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import CategoricalEncoder

# Making pipelines
numerical_pipeline = Pipeline([
    ("select_numeric", DataFrameSelector(["age", "balance", "day", "campaign", "pdays", "previous","duration"])),
    ("std_scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("select_cat", DataFrameSelector(["job", "education", "marital", "default", "housing", "loan", "contact", "month",
                                     "poutcome"])),
    ("cat_encoder", CategoricalEncoder(encoding='onehot-dense'))
])

from sklearn.pipeline import FeatureUnion
preprocess_pipeline = FeatureUnion(transformer_list=[
        ("numerical_pipeline", numerical_pipeline),
        ("categorical_pipeline", categorical_pipeline),
    ])
        
X_train = preprocess_pipeline.fit_transform(train_data)
y_train = train_data['deposit']
y_test = test_data['deposit']

y_train = LabelEncoder().fit_transform(y_train)
y_test = LabelEncoder().fit_transform(y_test)
y_train_yes = (y_train == 1)

#%%
import time
#from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

dict_classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Nearest Neighbors": KNeighborsClassifier(),
    "Linear SVM": SVC(),
    "Gradient Boosting Classifier": GradientBoostingClassifier(),
    "Decision Tree": tree.DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=18),
    "Neural Net": MLPClassifier(alpha=1),
    "Naive Bayes": GaussianNB()
}

len_classifiers = len(dict_classifiers.keys())

def batch_classify(X_train, Y_train, dict_classifiers, verbose = True):
    df_results = pd.DataFrame(np.zeros(shape=(len(dict_classifiers.keys()),3)), 
                              columns = ['classifier', 'train_score', 'training_time'])
    count = 0
    for key, classifier in dict_classifiers.items():
        t_start = time.clock()
        classifier.fit(X_train, Y_train)
        t_end = time.clock()
        t_diff = t_end - t_start
        train_score = classifier.score(X_train, Y_train) # .score() 內部會自行預測，不用先.predict(0)再用.score()
        #https://www.kaggle.com/getting-started/27261
        df_results.loc[count,'classifier'] = key
        df_results.loc[count,'train_score'] = train_score
        df_results.loc[count,'training_time'] = t_diff
        if verbose:
            print("trained {c} in {f:.2f} s".format(c=key, f=t_diff))
        count+=1
    return df_results

df_results = batch_classify(X_train, y_train, dict_classifiers)
print(df_results.sort_values(by='train_score', ascending=False))
# =>Decision Tree、Random Forest 有可能有overfitting=> 用交叉驗證
#https://blog.csdn.net/weixin_38536057/article/details/78702564

from sklearn.model_selection import cross_val_score

score_cv=[]
for key, classifier in dict_classifiers.items():
    scores_mean=cross_val_score(classifier, X_train, y_train, cv=3).mean()
    score_cv.append(scores_mean)

df_results['cv_mean_score']=score_cv
df_results.sort_values(by=['cv_mean_score'], ascending=False)
df_results.to_csv('df_results_models.csv')
# => Neural Net、Gradient Boosting Classifier、 Linear SVM 是前三高

#%%
'''
混淆矩陣
https://kknews.cc/zh-tw/news/jlm8pm6.html
'''
from sklearn.model_selection import cross_val_predict
# Gradient Boosting Classifier的training_score最高
grad_clf=GradientBoostingClassifier()
y_train_pred = cross_val_predict(grad_clf, X_train, y_train, cv=3)

from sklearn.metrics import accuracy_score
grad_clf.fit(X_train, y_train)
print ("Gradient Boost Classifier accuracy is %2.2f" % accuracy_score(y_train, y_train_pred))

from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_train, y_train_pred)
plt.rcParams.update({'font.size': 30}) # set font size
f, ax = plt.subplots(figsize=(20, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", linewidths=.5, ax=ax)
plt.title("Confusion Matrix", fontsize=32)
plt.subplots_adjust(left=0.15, right=0.99, bottom=0.15, top=0.8)
ax.set_yticks(np.arange(conf_matrix.shape[0]) + 0.5, minor=False)
ax.set_xticklabels(['Predict: Negative', 'Predict: Positive'], fontsize=25)
ax.set_yticklabels(['Actual: Negative', 'Actual: Positive'], fontsize=25, rotation=360)
plt.savefig('confusion_matrix1.png', dpi=300) # predicted(width) vs. actual(hight)
#plt.show()

from sklearn.metrics import precision_score, recall_score
print('Precision Score: ', precision_score(y_train, y_train_pred)) #0.824373576309795
print('Recall Score: ', recall_score(y_train, y_train_pred)) #0.8551512287334594

from sklearn.metrics import f1_score
f1_score(y_train, y_train_pred) #0.8394803989793551

#%%
'''
Recall Precision Tradeoff
https://www.twblogs.net/a/5d04f09cbd9eee47d34bd862
用decision_function函數計算決策分數，再依其設置threshold
'''
# 返回決策分數
y_scores = cross_val_predict(grad_clf, X_train, y_train, cv=3, method="predict_proba")
neural_y_scores = cross_val_predict(MLPClassifier(alpha=1), X_train, y_train, cv=3, method="predict_proba")
svc_y_scores = cross_val_predict(SVC(probability=True), X_train, y_train, cv=3, method="predict_proba")

if y_scores.ndim == 2: # 如果是二維
    y_scores = y_scores[:, 1]
if neural_y_scores.ndim == 2:
    neural_y_scores = neural_y_scores[:, 1]    
if svc_y_scores.ndim == 2:
    svc_y_scores = svc_y_scores[:, 1]

# How can we decide which threshold to use? We want to return the scores instead of predictions with this code.
from sklearn.metrics import precision_recall_curve

precisions, recalls, threshold = precision_recall_curve(y_train, y_scores)

def plot_precision_recall_vs_threshold(precisions, recalls, threshold):
    plt.rcParams.update({'font.size': 10}) # set font size
    plt.plot(threshold, precisions[:-1], "r--", label="Precision", linewidth=2)
    plt.plot(threshold, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.xlabel("Threshold", fontsize=16)
    plt.ylabel('Level of Precision and Recall', fontsize=16)
    plt.title('Precision and Recall \n Tradeoff', fontsize=18)
    plt.legend(loc="best", fontsize=10)
    plt.ylim([0, 1])

plt.figure(figsize=(10,5))
plot_precision_recall_vs_threshold(precisions, recalls, threshold)
plt.savefig('precision_recall_tradeoff1.png', dpi=300)

# get best tradeoff value
pr_df=pd.DataFrame(precisions, columns=['precisions'])
pr_df['recalls']=recalls
pr_best=pr_df[pr_df.precisions == pr_df.recalls]
pr_best_index=pr_best.index.values
pr_best_value=pr_best['recalls'].values #0.83135955
pr_best_threshold=threshold[pr_best_index].astype(float) #0.54426586

#%%
'''
ROC Curve (Receiver Operating Characteristic):
'''
from sklearn.metrics import roc_auc_score

print('Gradient Boost Classifier Score: ', roc_auc_score(y_train, y_scores)) #0.9120703947686892
print('Neural Classifier Score: ', roc_auc_score(y_train, neural_y_scores)) 
print('Naives Bayes Classifier: ', roc_auc_score(y_train, svc_y_scores)) 

from sklearn.metrics import roc_curve

grd_fpr, grd_tpr, grd_threshold = roc_curve(y_train, y_scores)
neu_fpr, neu_tpr, neu_threshold = roc_curve(y_train, neural_y_scores)
svc_fpr, svc_tpr, svc_threshold = roc_curve(y_train, svc_y_scores)

def graph_roc_curve(false_positive_rate, true_positive_rate, label=None):
    plt.figure(figsize=(10,6))
    plt.title('ROC Curve \n Gradient Boosting Classifier', fontsize=18)
    plt.plot(false_positive_rate, true_positive_rate, label=label)
    plt.plot([0, 1], [0, 1], '#0C8EE0')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    
graph_roc_curve(grd_fpr, grd_tpr, grd_threshold)
plt.savefig('roc_curve1.png', dpi=300)

#%%
'''
repredict
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
'''
data=pd.read_csv('bank.csv') # raw_data
data_copy=data.copy()

data_columns=['job', 'marital', 'education', 'default', 'housing','loan', 'contact', 
 'month', 'poutcome', 'deposit']
for i in data_columns:
    data_copy[i] = data_copy[i].astype('category').cat.codes # 轉成數字

y=data_copy['deposit']
data_copy=data_copy.drop('deposit', axis=1)
X=data_copy

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


from sklearn.model_selection import cross_val_predict

grad_clf=GradientBoostingClassifier()

y_train_roc_pred = (grad_clf.predict_proba(X_train)[:,1] > pr_best_threshold)# set threshold as 0.54426586
y_test_roc_pred = (grad_clf.predict_proba(X_test)[:,1] > pr_best_threshold) # set threshold as 0.54426586

from sklearn.metrics import accuracy_score
grad_clf.fit(X_train, y_train)
print ("Gradient Boost Classifier accuracy is %2.2f" % accuracy_score(y_train, y_train_roc_pred)) # 0.86
print ("Gradient Boost Classifier accuracy is %2.2f" % accuracy_score(y_test, y_test_roc_pred)) # 0.82

from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_train, y_train_roc_pred)
plt.rcParams.update({'font.size': 30}) # set font size
f, ax = plt.subplots(figsize=(20, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", linewidths=.5, ax=ax)
plt.title("Confusion Matrix", fontsize=32)
plt.subplots_adjust(left=0.15, right=0.99, bottom=0.15, top=0.8)
ax.set_yticks(np.arange(conf_matrix.shape[0]) + 0.5, minor=False)
ax.set_xticklabels(['Predict: Negative', 'Predict: Positive'], fontsize=25)
ax.set_yticklabels(['Actual: Negative', 'Actual: Positive'], fontsize=25, rotation=360)
plt.savefig('confusion_matrix4.png', dpi=300) # predicted(width) vs. actual(hight)
#plt.show()

from sklearn.metrics import precision_score, recall_score
print('Precision Score: ', precision_score(y_train, y_train_roc_pred)) #0.8472906403940886
print('Recall Score: ', recall_score(y_train, y_train_roc_pred)) #0.8555187115111321

from sklearn.metrics import f1_score
f1_score(y_train, y_train_roc_pred) #0.851384796700059

from sklearn.metrics import roc_auc_score
print('Gradient Boost Classifier ROC Score is %2.2f'% roc_auc_score(y_train, y_train_roc_pred)) #0.86
print('Gradient Boost Classifier ROC Score is %2.2f'% roc_auc_score(y_test, y_test_roc_pred)) #0.82

#%%
'''
feature importance
'''
def feature_importance_graph(indices, importances, feature_names):
    plt.figure(figsize=(12,6))
    plt.title("Determining Feature importances \n with Gradient Boost Classifier", fontsize=18)
    plt.barh(range(len(indices)), importances[indices], color='#FF7744',  align="center")
    plt.yticks(range(len(indices)), feature_names[indices], rotation='horizontal',fontsize=14)
    plt.ylim([-1, len(indices)])
    #plt.axhline(y=1.85, xmin=0.21, xmax=0.952, color='k', linewidth=3, linestyle='--')
    #plt.text(0.30, 2.8, '46% Difference between \n duration and contacts', color='k', fontsize=15)


grad_clf.fit(X_train, y_train)
importances=grad_clf.feature_importances_

feature_names = data_copy.columns 
indices = np.argsort(importances)[::-1]
feature_importance_graph(indices, importances, feature_names)
plt.savefig('feature_importance_gb2.png', dpi=300)

'''
most important features:
Duration (how long it took the conversation between the sales representative and the potential client), 
contact (number of contacts to the potential client within the same marketing campaign), 
month (the month of the year).
'''



