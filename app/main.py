from typing import Annotated
import io
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, Form
from modules.model import load_model, load_files, load_nn_model, preprocess_image, extract_features, retrieve_images, format_images

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir todos los orígenes
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model()
train_features_flat, train_labels_flat, train_images_flat = load_files()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/images/")
async def post_images(image: Annotated[UploadFile, Form()],
                      neighbors: Annotated[str, Form()]):

    nn_model = load_nn_model(train_features_flat, int(neighbors))
    query_img = Image.open(io.BytesIO(image.file.read()))

    processed_query = preprocess_image(query_img)
    query_features = extract_features(processed_query, model)

    images, labels = retrieve_images(
        query_features, nn_model, train_images_flat, train_labels_flat)

    labeled_images = format_images(images, labels)

    return {"images": labeled_images}
