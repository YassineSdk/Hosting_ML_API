import numpy as np
import pickle
from fastapi import FastAPI,File,UploadFile,HTTPException
from fastapi.middleware.cors import CORSMiddleware
import io 
import uvicorn
from  PIL import ImageOps ,Image


with open('mnist_model.pkl','rb') as f:
    model = pickle.load(f)


app = FastAPI(title="APP for hand written didgits Recognition ",
            description="""
            Upload an image of a handwritten digit (0–9)  
            and let the trained model predict what it is.  
            - Accepts `.png`, `.jpg`, `.jpeg`  
            - Output: predicted digit
            """,
            version="1.0.0",
            contact={
                "name": "Count Yassine",
                "url": "https://github.com/CountYassine",
                "email": "count@datascience.ma",
            },
)


#conditions
MAX_FILE_SIZE = 1 * 1024 * 1024
allowed_extentions = {'.png', '.gpl', '.ngp', '.gpn'}



def allowed_file(filename:str) -> bool:
    return any(filename.lower().endswith(ext) for ext in allowed_extentions)



@app.post("/predict_image")
async def predict_image(File:UploadFile = File(...)):

    if not allowed_file(File.filename):
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Allowed extensions: {', '.join(allowed_extentions)}"
        )
    contents = await File.read()
    if len(contents) > MAX_FILE_SIZE:
        return {"error":"file too large,Max size: 1Mb"}
    pil_image = Image.open(io.BytesIO(contents)).convert('L')
    pil_image = ImageOps.invert(pil_image)
    pil_image = pil_image.resize(size=(28,28))
    image_array = np.array(pil_image).reshape(1,-1)
    prediction = model.predict(image_array)
    return {"prediction" : int(prediction[0])}




if __name__ == "__main__":
    uvicorn.run("main:app",host="127.0.0.1",port=8000,reload=True)






