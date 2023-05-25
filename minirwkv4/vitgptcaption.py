from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import yaml
import torch

def readcog(path):
    with open(path, 'r',encoding='UTF-8') as file:
        data = file.read()
        result = yaml.safe_load(data)
        return result


LMyamlres = readcog("./config/minirwkv4.yaml")
model_path = LMyamlres['model-visual-caption']['Vpath']


model = VisionEncoderDecoderModel.from_pretrained(model_path)
feature_extractor = ViTImageProcessor.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length = 30
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

def get_vitgptcap(imgpath):
  raw_image = Image.open(imgpath).convert('RGB')

  pixel_values = feature_extractor(images=raw_image, return_tensors="pt").pixel_values
  pixel_values = pixel_values.to(device)

  output_ids = model.generate(pixel_values, **gen_kwargs)

  preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
  vcaption = [pred.strip() for pred in preds]

  return vcaption[0]

