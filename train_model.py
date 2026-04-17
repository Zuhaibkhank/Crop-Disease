import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

# =========================
# SETTINGS
# =========================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
DATASET_PATH = "dataset"

# =========================
# CHECK DATASET
# =========================
if not os.path.exists(DATASET_PATH):
    raise Exception("❌ 'dataset' folder not found!")

print("✅ Dataset found")

# =========================
# DATA AUGMENTATION (🔥 IMPROVES ACCURACY)
# =========================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

# =========================
# LOAD TRAIN DATA
# =========================
train_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

# =========================
# LOAD VALIDATION DATA
# =========================
val_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# =========================
# CLASS NAMES
# =========================
class_names = list(train_data.class_indices.keys())

print("\n📌 Classes Detected:")
for i, name in enumerate(class_names):
    print(f"{i} → {name}")

# =========================
# MODEL (IMPROVED)
# =========================
model = models.Sequential([

    layers.Input(shape=(224,224,3)),

    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(len(class_names), activation='softmax')
])

# =========================
# COMPILE
# =========================
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# =========================
# CALLBACKS (🔥 IMPORTANT)
# =========================
os.makedirs("model", exist_ok=True)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "model/crop_model.keras",
    monitor='val_accuracy',
    save_best_only=True
)

# =========================
# TRAIN
# =========================
print("\n🚀 Training Started...\n")

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint]
)

# =========================
# SAVE CLASS LABELS
# =========================
with open("model/classes.txt", "w") as f:
    for name in class_names:
        f.write(name + "\n")

model.save("model/crop_model.keras")

print("\n✅ Training Complete!")
print("📁 Best Model saved in 'model/crop_model.keras'")