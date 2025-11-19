import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

# create model directory if not exists
os.makedirs("model", exist_ok=True)

# ------------------------------
# STEP 1: Load Dataset
# ------------------------------
df = pd.read_csv(r"dataset/symptom_disease_dataset_3000(1).csv")

# merge all 8 symptoms into one text line
df["all_symptoms"] = df[
    ["symptom_1","symptom_2","symptom_3","symptom_4",
     "symptom_5","symptom_6","symptom_7","symptom_8"]
].astype(str).agg(" ".join, axis=1)

df = df[["all_symptoms", "disease"]]

# ------------------------------
# STEP 2: TF-IDF
# ------------------------------
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df["all_symptoms"])

# ------------------------------
# STEP 3: Label Encoding
# ------------------------------
le = LabelEncoder()
y = le.fit_transform(df["disease"])

# ------------------------------
# STEP 4: Train-test split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------
# STEP 5: Build Neural Network
# ------------------------------
model = models.Sequential()
model.add(layers.Input(shape=(X_train.shape[1],)))
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(len(le.classes_), activation="softmax"))

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ------------------------------
# STEP 6: Train Model
# ------------------------------
history = model.fit(
    X_train.toarray(),
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test.toarray(), y_test)
)

# ------------------------------
# STEP 7: Save TF-IDF, LabelEncoder, and Model
# ------------------------------
joblib.dump(tfidf, "model/tfidf.pkl")
joblib.dump(le, "model/label_encoder.pkl")
model.save("model/nn_model.h5")

print("\nModel saved successfully!")
print("Files created:")
print(" - model/tfidf.pkl")
print(" - model/label_encoder.pkl")
print(" - model/nn_model.h5")
