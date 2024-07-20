from ai import ImageGeneratorPipeline
from PIL import Image, ImageFilter
import numpy as np

class ImagePipelineProcessor:
  def __init__(self, text, padding=40, blur_radius=12):
    self.text = text
    self.padding = padding
    self.blur_radius = blur_radius
    self.pipeline = ImageGeneratorPipeline()

  def generate_images(self):
    image, depth = self.pipeline.generate_images(self.text)
    return image, depth

  def crop_images(self, image, depth):
    return (image.crop((self.padding, self.padding, image.width - self.padding, image.height - self.padding)),
            depth.crop((self.padding, self.padding, depth.width - self.padding, depth.height - self.padding)))

  def process_depth_image(self, depth):
    depth = depth.filter(ImageFilter.GaussianBlur(self.blur_radius))
    depth_array = np.array(depth)
    depth_array = (depth_array - np.min(depth_array)) / (np.max(depth_array) - np.min(depth_array))  # Normalize
    return self.quantize_depth(depth_array)

  def quantize_depth(self, depth_array, threshold=0.5):
    depth_array[depth_array < threshold] = 0
    depth_array[depth_array >= threshold] = 1
    return depth_array

  def segment_image(self, image, depth_array):
    foreground = np.array(image).copy()
    background = np.array(image).copy()
    foreground[depth_array != 0] = 0
    background[depth_array == 0] = 0
    return foreground, background

  def save_foreground(self, foreground):
    foreground_image = Image.fromarray(foreground).convert("RGBA")
    data = foreground_image.getdata()
    new_data = [(0, 0, 0, 0) if item[0] == 0 and item[1] == 0 and item[2] == 0 else item for item in data]
    foreground_image.putdata(new_data)
    foreground_image.save("foreground.png", format="PNG")

  def save_background(self, image):
    blurred_image = image.filter(ImageFilter.GaussianBlur(10))
    blurred_image.save("background.png", format="PNG")

  def run(self):
    image, depth = self.generate_images()
    image, depth = self.crop_images(image, depth)
    depth_array = self.process_depth_image(depth)
    foreground, background = self.segment_image(image, depth_array)
    self.save_foreground(foreground)
    self.save_background(image)

if __name__ == "__main__":
    processor = ImagePipelineProcessor("A knight fighting a dragon, highly detailed, beautiful and colorful picture")
    processor.run()
