from PIL import Image
import numpy as np
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf
import io
import base64

target_size = (170, 170)  # Tamaño deseado para redimensionamiento


def preprocess_image(query_img):
    if query_img.mode != 'RGB':
        query_img = query_img.convert('RGB')

    # Redimensionar imagen a un tamaño fijo usando LANCZOS
    img_resized = query_img.resize(target_size, Image.LANCZOS)

    # Convertir imagen a un array de NumPy
    img_array = np.array(img_resized, dtype='float32')
    img_array = preprocess_input(img_array)

    processed_query = tf.expand_dims(img_array, axis=0)

    return processed_query


def load_model():
    base_model = VGG19(weights='imagenet', include_top=False,
                       input_shape=(target_size[0], target_size[1], 3))

    model = Model(inputs=base_model.input,
                  outputs=base_model.layers[-1].output)
    return model


def load_files():
    train_features_flat = np.load("app/data/train_features_flat.npy")
    train_labels_flat = np.load("app/data/train_labels_flat.npy")
    train_images_flat = np.load("app/data/train_imgs_flat.npy",)

    return train_features_flat, train_labels_flat, train_images_flat


def load_nn_model(train_features_flat, neighbors=5):
    nn_model = NearestNeighbors(
        n_neighbors=neighbors, algorithm='ball_tree').fit(train_features_flat)
    return nn_model


def extract_features(processed_query, model):
    query_features = model.predict([processed_query]).flatten().reshape(1, -1)

    return query_features


def denormalize_image(img):
    mean = [103.939, 116.779, 123.68]  # Media para VGG19
    std = [1, 1, 1]  # Desviación estándar (simplificado para este ejemplo)

    # Deshacer la normalización
    img_denormalized = img * std + mean

    # Convertir de BGR a RGB (si es necesario)
    img_denormalized = img_denormalized[..., ::-1]

    # Asegurar que los valores estén entre 0 y 255
    img_denormalized = np.clip(img_denormalized, 0, 255).astype(np.uint8)
    return img_denormalized


def retrieve_images(query_features, nn_model, train_images_flat,
                    train_labels_flat):
    retrieved_images = []
    retrieved_labels = []
    distances, indices = nn_model.kneighbors(query_features)

    for indice in indices.flatten():
        retrieved_images.append(denormalize_image(train_images_flat[indice]))
        retrieved_labels.append(train_labels_flat[indice])

    return retrieved_images, retrieved_labels


def format_images(images: list, labels: list) -> list:
    labeled_images = []

    for index, image in enumerate(images):
        retrieved_image = Image.fromarray(image.astype("uint8"))

        image_bytes = io.BytesIO()

        retrieved_image.save(image_bytes, format="JPEG")

        image_string = base64.b64encode(image_bytes.getvalue()).decode("utf-8")

        base64_image = f"data:image/jpeg;base64,{image_string}"

        labeled_images.append({"source": base64_image, "label": labels[index]})

    return labeled_images
