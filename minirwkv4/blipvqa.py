from PIL import Image
from transformers import  BlipProcessor, BlipForQuestionAnswering
import yaml
import torch

def readcog(path):
    with open(path, 'r',encoding='UTF-8') as file:
        data = file.read()
        result = yaml.safe_load(data)
        return result


LMyamlres = readcog("./config/minirwkv4.yaml")
model_path = LMyamlres['model-visual-qa']['Bpath']


processor = BlipProcessor.from_pretrained(model_path)
model = BlipForQuestionAnswering.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def get_bqares(imgpath,text):
    raw_image = Image.open(imgpath).convert('RGB')
    inputs = processor(raw_image, text, return_tensors="pt").to("cuda")
    out = model.generate(**inputs)
    vqares = processor.decode(out[0], skip_special_tokens=True)
    return vqares



