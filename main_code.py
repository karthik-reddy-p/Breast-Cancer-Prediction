import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adadelta  # Import Adadelta optimizer
import matplotlib.pyplot as plt

# Mixed precision
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

def load_images_from_folder(dataset_path, img_size=(256, 256)):
    images, masks = [], []
    categories = ['benign', 'normal', 'malignant']

    for category in categories:
        category_path = os.path.join(dataset_path, category)
        image_files = sorted([f for f in os.listdir(category_path) if "mask" not in f.lower()])
        mask_files = sorted([f for f in os.listdir(category_path) if "mask" in f.lower()])

        for img_file, mask_file in zip(image_files, mask_files):
            img_path, mask_path = os.path.join(category_path, img_file), os.path.join(category_path, mask_file)
            img, mask = cv2.imread(img_path), cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if img is not None and mask is not None:
                images.append(cv2.resize(img, img_size) / 255.0)
                masks.append(cv2.resize(mask, img_size) / 255.0)

    return np.array(images), np.expand_dims(np.array(masks), axis=-1)

dataset_path = "/content/drive/MyDrive/Colab Notebooks/Dataset_BUSI_with_GT"
train_images, train_masks = load_images_from_folder(dataset_path, img_size=(128,128)) #reduce image size
test_images, test_masks = load_images_from_folder(dataset_path, img_size=(128,128))

print("Train Images Shape:", train_images.shape)
print("Train Masks Shape:", train_masks.shape)
print("Test Images Shape:", test_images.shape)
print("Test Masks Shape:", test_masks.shape)

def build_model(input_shape=(128, 128, 3)): #reduced input shape
    inputs = Input(input_shape)
    base_model = EfficientNetB0(include_top=False, weights="imagenet", input_tensor=inputs)

    encoder_output = base_model.output

    # Decoder (Upsampling Layers) - reduced filters
    x = Conv2DTranspose(128, (3,3), strides=(2,2), padding="same")(encoder_output)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2DTranspose(64, (3,3), strides=(2,2), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2DTranspose(32, (3,3), strides=(2,2), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2DTranspose(1, (3,3), strides=(2,2), padding='same')(x)
    x = Activation('sigmoid')(x)

    model = Model(inputs, x)
    return model

unet_model = build_model()
unet_model.compile(optimizer=Adadelta(), loss="binary_crossentropy", metrics=["accuracy"]) # Use Adadelta
unet_model.summary()

history = unet_model.fit(
    train_images, train_masks,
    validation_data=(test_images, test_masks),
    epochs=50,
    batch_size=32, #increased batch size
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
)

def plot_sample(image, predicted_mask, actual_mask, index):
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    ax[0].imshow(image, cmap='gray')
    ax[0].set_title(f'Original Image {index}')
    ax[0].axis('off')

    overlay_predicted = image.copy()
    overlay_predicted[predicted_mask.squeeze() > 0.5] = [0, 0, 255]
    ax[1].imshow(overlay_predicted)
    ax[1].set_title(f'Predicted Mask (Blue) {index}')
    ax[1].axis('off')

    overlay_actual = image.copy()
    overlay_actual[actual_mask.squeeze() > 0.5] = [0, 0, 255]
    ax[2].imshow(overlay_actual)
    ax[2].set_title(f'Actual Mask (Blue) {index}')
    ax[2].axis('off')

    plt.show()

predictions = unet_model.predict(test_images)
predictions = (predictions > 0.5).astype(np.uint8)

for i in range(3):
    plot_sample(test_images[i], predictions[i].squeeze(), test_masks[i].squeeze(), i)
unet_model.save('unet_model.keras')
