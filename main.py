from fastapi import FastAPI,status , UploadFile, File
from api import predictBefore
import keras
from contextlib import asynccontextmanager
import os
from ultralytics import YOLO
MODEL_PATH = {
    'classificationMenuModel' : 'weights/before/best.pt',
    'croppedAfter' : 'weights/after/cropBest.pt',
    'regressionAfter' : 'weights/after/best_model.keras',

}



ml_models = {}
@asynccontextmanager
async def lifespan(app: FastAPI):
    print('Loading AI Model')
    for modelName, modelPath in MODEL_PATH.items():
        try:
            if modelPath.endswith('pt'):
                print(f'Loading Model: {modelName} Path: {modelPath}')
                ml_models[modelName] = YOLO(modelPath)
                print(f'Successed Loading Model: {modelName}')
            elif modelPath.endswith('.keras'):
                print(f'Loading Model: {modelName} Path: {modelPath}')
                ml_models[modelName] = keras.models.load_model(modelPath)
                print(f'Successed Loading Model: {modelName}')

        except Exception as e:
            print(f'Can not load Model: {modelName}')

    yield {'models' : ml_models}
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

app.include_router(predictBefore.router)


