import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
import joblib
import os

# Đường dẫn file dữ liệu
DATA_PATH = 'data/data_youtube_trending_video.csv'
MODEL_PATH = 'xgb_trending_model.pkl'
PIPELINE_PATH = 'xgb_pipeline.pkl'

# Đọc dữ liệu
print('Đang đọc dữ liệu...')
df = pd.read_csv(DATA_PATH)

# Xử lý các trường cần thiết
features = [
    'title', 'description', 'tags', 'categoryId', 'viewCount', 'likeCount', 'commentCount'
]
target = 'isTrending'

# Xử lý missing
for col in features:
    if col not in df.columns:
        raise Exception(f'Missing column: {col}')

df = df.dropna(subset=features + [target])

# Chuyển target về 0/1
if df[target].dtype != int:
    df[target] = df[target].astype(str).str.strip().replace({'True': 1, 'False': 0, '1': 1, '0': 0})
    df[target] = df[target].astype(int)

# Tiền xử lý text: độ dài, số từ, số tag
print('Tiền xử lý text...')
def text_length(s):
    return len(str(s))
def word_count(s):
    return len(str(s).split())
def tag_count(s):
    return len(str(s).split(','))

df['title_len'] = df['title'].apply(text_length)
df['desc_len'] = df['description'].apply(text_length)
df['desc_words'] = df['description'].apply(word_count)
df['tag_count'] = df['tags'].apply(tag_count)

# Chuyển categoryId về số
if df['categoryId'].dtype == object:
    df['categoryId'] = LabelEncoder().fit_transform(df['categoryId'].astype(str))

# Chuyển các trường số về float
for col in ['viewCount', 'likeCount', 'commentCount']:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# Chọn feature cuối cùng
X = df[['title_len', 'desc_len', 'desc_words', 'tag_count', 'categoryId', 'viewCount', 'likeCount', 'commentCount']]
y = df[target]

# Chia train/test
print('Chia dữ liệu train/test...')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Xử lý mất cân bằng bằng sample_weight
sample_weight = compute_sample_weight(class_weight='balanced', y=y_train)

# Chuẩn hóa
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Huấn luyện XGBoost
print('Huấn luyện mô hình XGBoost...')
model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
model.fit(X_train_scaled, y_train, sample_weight=sample_weight)

# Đánh giá
score = model.score(X_test_scaled, y_test)
print(f'Độ chính xác trên tập test: {score:.4f}')

# Lưu model và pipeline
print('Lưu model và pipeline...')
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, PIPELINE_PATH)
print(f'Model đã lưu tại {MODEL_PATH}, pipeline tại {PIPELINE_PATH}') 