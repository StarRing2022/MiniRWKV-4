from PIL import Image
from transformers import  ViltProcessor, ViltForQuestionAnswering
import yaml
import torch

def readcog(path):
    with open(path, 'r',encoding='UTF-8') as file:
        data = file.read()
        result = yaml.safe_load(data)
        return result


LMyamlres = readcog("./config/minirwkv4.yaml")
model_path = LMyamlres['model-visual-qa']['Vpath']


processor = ViltProcessor.from_pretrained(model_path)
model = ViltForQuestionAnswering.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def get_vqares(imgpath,text):
    raw_image = Image.open(imgpath).convert('RGB')
    encoding = processor(raw_image, text, return_tensors="pt").to(device)
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    vqares = model.config.id2label[idx]
    return vqares


