import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =======================
# Parameters
# =======================
img_size = 224
data_dir = "data/"   # Folder with subfolders 'Benign' and 'Malignant'
model_save_path = "models/cnn_model.h5"

os.makedirs("models", exist_ok=True)

# =======================
# Data generators
# =======================
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(img_size, img_size),
    class_mode="binary",
    subset="training",
    shuffle=True,
    batch_size=32
)

val_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(img_size, img_size),
    class_mode="binary",
    subset="validation",
    shuffle=False,
    batch_size=32
)

# =======================
# CNN Model
# =======================
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(img_size, img_size, 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# =======================
# Train Model
# =======================
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=5
)

# =======================
# Save model
# =======================
model.save(model_save_path)
print("✅ CNN model saved at", model_save_path)

# =======================
# Predictions on Validation
# =======================
val_gen.reset()
y_pred_prob = model.predict(val_gen)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()
y_true = val_gen.classes

# =======================
# Confusion Matrix & Metrics
# =======================
cm = confusion_matrix(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("\n=== Confusion Matrix ===")
print(cm)
print("\nAccuracy  :", round(accuracy*100, 2), "%")
print("Precision :", round(precision*100, 2), "%")
print("Recall    :", round(recall*100, 2), "%")
print("F1 Score  :", round(f1*100, 2), "%")

# =======================
# Plot Confusion Matrix
# =======================
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign','Malignant'], yticklabels=['Benign','Malignant'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()