import requests
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
from io import BytesIO
from pydantic import BaseModel, Field
from typing import List, Optional
import inferless

@inferless.request
class RequestObjects(BaseModel):
    image_url: str = Field(default="https://raw.githubusercontent.com/HistAI/hibou/refs/heads/main/images/sample.png")

@inferless.response
class ResponseObjects(BaseModel):
    outputs: List[float] = Field(default=[-0.8563,  2.3687])


class InferlessPythonModel:

    def initialize(self):
        self.processor = AutoImageProcessor.from_pretrained("histai/hibou-L", trust_remote_code=True)
        self.model = AutoModel.from_pretrained("histai/hibou-L", trust_remote_code=True)

    def infer(self, inputs):
        response = requests.get(inputs["image_url"])
        image = Image.open(BytesIO(response.content))
        
        inputs = self.processor(images=image, return_tensors="pt")
        
        # Perform inference
        outputs = self.model(**inputs)
        
        return {"outputs": outputs }

    # perform any cleanup activity here
    def finalize(self,args):
        self.pipe = None
