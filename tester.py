import img_preprocessor
from PIL import Image

def main():
  preproc = img_preprocessor.ImgPreprocessor()
  img = Image.open("image.jpg").convert('RGB')
  preproc.get_image_parse(img)

if __name__ == "__main__":
  main()