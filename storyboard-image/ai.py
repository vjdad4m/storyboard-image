import torch
from diffusers import DiffusionPipeline
from transformers import AutoFeatureExtractor, pipeline
import numpy as np
from PIL import Image

def load_pipeline(model_name, device):
  pipeline = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
  return pipeline.to(device)

def generate_image(pipeline, text, height, width):
  pipeline.scheduler.set_timesteps(50)
  return pipeline(text, height=height, width=width).images[0]

def save_image(image, filename):
  image.save(filename)

def load_depth_estimator(model_name, device):
  depth_estimator = pipeline("depth-estimation", model=model_name)
  depth_estimator.model = depth_estimator.model.to(device).half()
  return depth_estimator

def preprocess_image(image, feature_extractor, device):
  inputs = feature_extractor(images=image, return_tensors="pt")
  return {k: v.to(device).half() for k, v in inputs.items()}

def estimate_depth(depth_estimator, inputs):
  with torch.no_grad():
    outputs = depth_estimator.model(**inputs)
  return outputs.predicted_depth

def save_depth_image(prediction, filename):
  prediction_image = prediction.squeeze().cpu().numpy()
  depth_image = Image.fromarray((prediction_image * 255 / prediction_image.max()).astype(np.uint8))
  depth_image.save(filename)

class ImageGeneratorPipeline:
  def __init__(self):
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.pipeline = load_pipeline("stablediffusionapi/toonyou", self.device)
    self.depth_estimator = load_depth_estimator("vinvino02/glpn-nyu", self.device)
    self.feature_extractor = AutoFeatureExtractor.from_pretrained("vinvino02/glpn-nyu")
  
  def generate_images(self, text, height=1024, width=1024):
    image = generate_image(self.pipeline, text, height, width)
    inputs = preprocess_image(image, self.feature_extractor, self.device)
    prediction = estimate_depth(self.depth_estimator, inputs)  
    prediction_image = prediction.squeeze().cpu().numpy()
    depth_image = Image.fromarray((prediction_image * 255 / prediction_image.max()).astype(np.uint8))
    return image, depth_image

def main():
  width, height = 1024, 1024
  device = "cuda" if torch.cuda.is_available() else "cpu"

  pipeline = load_pipeline("stablediffusionapi/toonyou", device)
  text = "A knight fighting a dragon, highly detailed, beautiful and colorful picture"
  image = generate_image(pipeline, text, height, width)
  save_image(image, "image.jpg")

  depth_estimator = load_depth_estimator("vinvino02/glpn-nyu", device)
  feature_extractor = AutoFeatureExtractor.from_pretrained("vinvino02/glpn-nyu")
  
  inputs = preprocess_image(image, feature_extractor, device)
  prediction = estimate_depth(depth_estimator, inputs)
  save_depth_image(prediction, "depth.jpg")

if __name__ == "__main__":
  main()
