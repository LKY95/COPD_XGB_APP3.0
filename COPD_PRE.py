import pandas as pd
import xgboost as xgb
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 定义特征类型
CAT_FEATURES = ['gender', 'smoke']  # 分类特征
FEATURES = [
    'f1', 'A1', 'r1', 'tt1', 'eta1', 'area1', 'nl11', 'nl12', 'nl13',
    'omega_mean11', 'omega_mean12', 'omega_mean13', 'f2', 'A2', 'r2',
    'tt2', 'eta2', 'area2', 'nl21', 'nl22', 'nl23', 'omega_mean21',
    'omega_mean22', 'omega_mean23', 'age', 'BMI', 'gender', 'smoke'
]


def preprocess_data(train, val):
    """ 数据预处理流程（返回scaler） """
    cont_features = [f for f in FEATURES if f not in CAT_FEATURES]
    scaler = StandardScaler()
    train[cont_features] = scaler.fit_transform(train[cont_features])
    val[cont_features] = scaler.transform(val[cont_features])
    return train, val, scaler  # 返回scaler


# 加载数据
train = pd.read_csv('5_1_Sit_Stand_COPD_NOCOPD_MI_8_2_Smote_NC_Training_Set.csv')
val = pd.read_csv('5_2_Sit_Stand_COPD_NOCOPD_MI_8_2_NO_Smote_NC_Validation_Set.csv')

# 执行预处理并获取scaler
train_preprocessed, val_preprocessed, scaler = preprocess_data(train.copy(), val.copy())  # 接收scaler

# 划分特征标签
X_train = train_preprocessed[FEATURES]
y_train = train_preprocessed['COPD']
X_val = val_preprocessed[FEATURES]
y_val = val_preprocessed['COPD']

# 强制转换分类变量类型（重要）
for df in [X_train, X_val]:
    df[CAT_FEATURES] = df[CAT_FEATURES].astype(int)

# 训练模型
model = xgb.XGBClassifier(
    objective='binary:logistic',
    use_label_encoder=False,
    eval_metric='logloss',
    learning_rate=0.05,  # 更保守的学习率
    max_depth=4,  # 减少过拟合风险
    subsample=0.8,
    colsample_bytree=0.7,
    n_estimators=1000  # 配合早停使用
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50,
    verbose=10
)

# 保存模型和标准化器
with open('copd_xgb_model_v2.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('standard_scaler.pkl', 'wb') as f:  # 新增scaler保存
    pickle.dump(scaler, f)

print("模型已保存为 copd_xgb_model_v2.pkl")
print("标准化器已保存为 standard_scaler.pkl")