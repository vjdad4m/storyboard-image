import gradio as gr
from pipeline import ImagePipelineProcessor
from PIL import Image

def process_image(text):
  processor = ImagePipelineProcessor(text)
  processor.run()
  
  foreground_image = Image.open("foreground.png")
  background_image = Image.open("background.png")
  
  return foreground_image, background_image

iface = gr.Interface(
  fn=process_image,
  inputs=gr.Textbox(label="Enter your prompt", placeholder="A knight fighting a dragon, highly detailed, beautiful and colorful picture"),
  outputs=[gr.Image(label="Cropped Foreground"), gr.Image(label="Blurred Background")],
  title="Image Segmentation with Depth Estimation",
  description="Enter a prompt to generate an image, and see the segmented foreground and blurred background."
)

if __name__ == "__main__":
  iface.launch()
