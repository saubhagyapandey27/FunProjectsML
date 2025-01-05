import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
from batchloader import DataGenerator

# Load data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# Update image paths
train["imagePath"] = "data/Train/" + train["imagePath"]
test["imagePath"] = "data/Test/" + test["imagePath"]

# Encode labels
train['label'] = train['label'].astype(str)
test['label'] = test['label'].astype(str)
encoder = OneHotEncoder(sparse_output=False)
train_labels = encoder.fit_transform(train[['label']])
test_labels = encoder.transform(test[['label']])

# Initialize generators
batch_size = 32
train_generator = DataGenerator(train["imagePath"].values, train_labels, batch_size)
val_generator = DataGenerator(test["imagePath"].values, test_labels, batch_size)

# Build model
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers[:-10]:
    layer.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_labels.shape[1], activation='softmax')
])

# Compile model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    verbose=1
)

# Evaluate model
test_loss, test_accuracy = model.evaluate(val_generator)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Save model
model.save('efficientnetb0_transfer_learning_with_batch_processing_finetuned.h5')