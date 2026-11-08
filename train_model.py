import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Set the paths for the dataset
train_dir = 'Plant_leave_diseases_dataset_with_augmentation\Train'  

# Check if the train directory exists
if not os.path.exists(train_dir):
    print(f"Error: The train directory '{train_dir}' does not exist!")
    exit()

# Set up ImageDataGenerator with rescaling and validation split
datagen = ImageDataGenerator(
    rescale=1./255,                  # Normalize pixel values to [0, 1]
    validation_split=0.2             # 80% train, 20% validation
)

# Flow training images from the 'train' directory
train_data = datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),         # Resize images to 128x128 pixels
    batch_size=16,
    class_mode='categorical',       # Categorical classification (since we have multiple classes)
    subset='training'               # This uses the 'training' split (80% of the data)
)

# Flow validation images from the 'train' directory
val_data = datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=16,
    class_mode='categorical',       # Categorical classification (same classes as training)
    subset='validation'             # This uses the 'validation' split (20% of the data)
)

# Build the model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(train_data.class_indices), activation='softmax')  # Number of classes based on train data
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # Loss function for multi-class classification
              metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model
history = model.fit(
    train_data, 
    validation_data=val_data,
    epochs=10  # Number of epochs (adjust as needed)
)

# Save the trained model
model.save('model/plant_disease_model.h5')  # Save the model in the 'model' directory

print("Model training complete and saved!")
