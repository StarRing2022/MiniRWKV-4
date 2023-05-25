from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import yaml
import torch

def readcog(path):
    with open(path, 'r',encoding='UTF-8') as file:
        data = file.read()
        result = yaml.safe_load(data)
        return result


LMyamlres = readcog("./config/minirwkv4.yaml")
model_path = LMyamlres['model-visual-caption']['Bpath']


processor = BlipProcessor.from_pretrained(model_path)
model = BlipForConditionalGeneration.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def get_blipcap(imgpath):
    raw_image = Image.open(imgpath).convert('RGB')
    inputs = processor(raw_image, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    vcaption = processor.decode(out[0], skip_special_tokens=True)
    return vcaption

