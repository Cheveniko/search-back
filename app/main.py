from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir todos los orígenes
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/images/")
async def retrieve_images(image: UploadFile):

    filename = image.filename
    content = await image.read()
    print(content)
    return {"image": image, "filename": filename}
