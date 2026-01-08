from fastapi import APIRouter, Request, UploadFile, File, HTTPException, status
from schemas.predictionBefore import ImagePredictionBefore, PredictionResponse
from PIL import Image
import io
import base64

router = APIRouter(
    prefix= "/before",
    tags = ["before"]
)

def image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    # Save เป็น JPEG เพื่อลดขนาดไฟล์ (ปรับ quality ได้ตามต้องการ)
    image.save(buffered, format="JPEG", quality=90)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")



@router.post('/predict', response_model=PredictionResponse,status_code=status.HTTP_200_OK)
async def predictMenu(request: Request,file: UploadFile = File(...)):
    try:
        model = request.state.models['classificationMenuModel']
        
        image_bytes = await file.read()
        originalImage = Image.open(io.BytesIO(image_bytes))

        results = model(originalImage)

        outputList = []

        for result in results:
            boxes = result.boxes

            for box in boxes:

                x1, y1, x2, y2 =map(int, box.xyxy[0].tolist())

                cls_id = int(box.cls[0])
                class_name = result.names[cls_id]
                conf = float(box.conf[0])

                cropped_img = originalImage.crop((x1, y1, x2, y2))
                img_base64 = image_to_base64(cropped_img)

                prediction = ImagePredictionBefore(
                    imageData = img_base64,
                    predictBowlMenu=class_name,
                    confidence=conf
                )
                outputList.append(prediction)

        return PredictionResponse(predictions = outputList)
    except Exception as e:
        print(f'Error {e}')
        raise HTTPException(status_code=500,detail=str(e))
