import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageChops, ImageEnhance
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ==========================================
# 1. CONFIGURATION
# ==========================================
IMAGE_SIZE = (128, 128)  # Smaller size = Faster CPU training
BATCH_SIZE = 32
EPOCHS = 20
DATASET_PATH = './dataset' # Path to your dataset folder

# ==========================================
# 2. ELA PREPROCESSING FUNCTION
# ==========================================
def convert_to_ela_image(path, quality=90):
    """
    Generates an Error Level Analysis (ELA) image.
    1. Saves the image at a specific quality (compression).
    2. Calculates the difference between original and compressed.
    3. Enhances the difference for visual clarity.
    """
    temp_filename = 'temp_ela.jpg'
    
    try:
        image = Image.open(path).convert('RGB')
        
        # Save compressed version to a temporary file
        image.save(temp_filename, 'JPEG', quality=quality)
        
        # Open compressed version
        compressed_image = Image.open(temp_filename)
        
        # Calculate the difference (ELA)
        ela_image = ImageChops.difference(image, compressed_image)
        
        # Enhance the ELA image brightness so we can see it
        extrema = ela_image.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        if max_diff == 0:
            max_diff = 1
        scale = 255.0 / max_diff
        
        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
        
        return ela_image.resize(IMAGE_SIZE)
        
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return None

# ==========================================
# 3. DATASET LOADING
# ==========================================
def load_dataset():
    X = []
    Y = []
    
    classes = ['real', 'forged'] # Folder names
    
    for label, class_name in enumerate(classes):
        folder_path = os.path.join(DATASET_PATH, class_name)
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} not found. Create it to train.")
            continue
            
        print(f"Processing {class_name} images...")
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            
            # Only process valid image files
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                ela_img = convert_to_ela_image(img_path)
                if ela_img:
                    X.append(np.array(ela_img) / 255.0) # Normalize pixel values
                    Y.append(label) # 0 for real, 1 for forged

    X = np.array(X)
    Y = to_categorical(Y, 2) # One-hot encoding
    return X, Y

# Load Data
print("Loading and preprocessing data (Computing ELA)...")
X, Y = load_dataset()

if len(X) == 0:
    print("No images found! Please check your dataset folder structure.")
    exit()

# Split into Train and Test
x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
print(f"Training samples: {len(x_train)}, Validation samples: {len(x_val)}")

# ==========================================
# 4. BUILD CNN MODEL
# ==========================================
model = Sequential([
    # Convolutional Block 1
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
    MaxPooling2D(pool_size=(2, 2)),
    
    # Convolutional Block 2
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    # Convolutional Block 3 (Optional for deeper feature extraction)
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    # Flatten and Dense Layers
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5), # Reduces overfitting
    Dense(2, activation='softmax') # Output: [Probability Real, Probability Forged]
])

model.compile(optimizer=Adam(learning_rate=0.001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

model.summary()

# ==========================================
# 5. TRAIN MODEL
# ==========================================
print("Starting training...")
history = model.fit(x_train, y_train, 
                    batch_size=BATCH_SIZE, 
                    epochs=EPOCHS, 
                    validation_data=(x_val, y_val),
                    verbose=1)

# ==========================================
# 6. EVALUATION & VISUALIZATION
# ==========================================

# Plot Training History
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Model Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.legend()
plt.show()

# Test on a specific image (Inference)
def predict_image(img_path):
    print(f"\nAnalyzing: {img_path}")
    original = Image.open(img_path).resize(IMAGE_SIZE)
    ela_img = convert_to_ela_image(img_path)
    
    # Prepare for model
    img_array = np.array(ela_img) / 255.0
    img_array = img_array.reshape(1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    
    prediction = model.predict(img_array)
    confidence = np.max(prediction)
    label = "Forged" if np.argmax(prediction) == 1 else "Real"
    
    # Visualization
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(ela_img)
    plt.title(f"ELA Input\nPrediction: {label} ({confidence:.2f})")
    plt.axis('off')
    
    plt.show()

# Example usage (Uncomment and replace path to test)
# predict_image('./dataset/forged/some_fake_image.jpg')