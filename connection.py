
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from pymongo import MongoClient
from datetime import datetime as time
import base64
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

app = FastAPI()

uri = "mongodb+srv://nikhil:eWBa6vxlpArO9jxw@deepfakedetection.vzkq85e.mongodb.net/?retryWrites=true&w=majority&appName=deepfakedetection"

client = MongoClient(uri, server_api=ServerApi('1'))

try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print("Error connecting to MongoDB:")
    print(e)

def upload_image(image_path, collection):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        document = {"image_name": image_path.split("/")[-1], "image_data": encoded_string , "time" : time.today()}
        collection.insert_one(document)
        print(f"Image {image_path} uploaded successfully.")

def retrieve_image(image_name, collection, output_path):
    document = collection.find_one({"image_name": image_name})
    if document:
        decoded_image = base64.b64decode(document["image_data"])
        with open(output_path, "wb") as output_file:
            output_file.write(decoded_image)
        print(f"Image {image_name} retrieved and saved to {output_path}.")
    else:
        print(f"Image {image_name} not found in the database.")

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    encoded_string = base64.b64encode(contents).decode('utf-8')
    document = {"image_name": file.filename, "image_data": encoded_string}
    image_collection.insert_one(document)
    return JSONResponse(content={"message": "File uploaded successfully."})

@app.get("/retrieve/{image_name}")
async def retrieve_file(image_name: str):
    db = client["deepfakedetection"]
    image_collection = db["userphotos"]
    document = image_collection.find_one({"image_name": image_name})
    if document:
        decoded_image = base64.b64decode(document["image_data"])
        return JSONResponse(content={"image_data": decoded_image})
    else:
        return JSONResponse(content={"message": "Image not found."}, status_code=404)

if __name__ == "__main__":
    db = client["deepfakedetection"]
    image_collection = db["userphotos"]

    upload_image("dog.jpg", image_collection)

    retrieve_image("dog.jpg", image_collection, "output_path.jpg") 