import os
import random
from data import get_image_list
from model import create_model
from data import read_img_path, tensor_to_img, save_image
import gradio as gr
import torchtext
from PIL import Image
import cv2
import os, os.path
import moviepy.video.io.ImageSequenceClip


torchtext.utils.download_from_url("https://drive.google.com/uc?id=1RILKwUdjjBBngB17JHwhZNBEaW4Mr-Ml", root="./weights/")
model = create_model()
os.makedirs("output", exist_ok=True)
os.makedirs("final", exist_ok=True)

def sketch2animevid(vid, load_size=512):
    vidcap = cv2.VideoCapture(vid)
    fps0 = vidcap.get(cv2.CAP_PROP_FPS)
    success,image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite("output/frame%d.jpg" % count, image)     # save frame as JPEG file      
        success,image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1
  

    counter = 0
    valid_images = [".jpg",".gif",".png",".tga"]
    for f in os.listdir('./output'):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        img, aus_resize = read_img_path(os.path.join("./output",f), load_size)
        aus_tensor = model(img)
        aus_img = tensor_to_img(aus_tensor)
        image_pil = Image.fromarray(aus_img)
        image_pil = image_pil.resize(aus_resize, Image.BICUBIC)
        image_pil.save(f'./final/{counter}.jpg')
        counter += 1

    
    image_folder='./final'
    fps=fps0

    image_files = [image_folder+'/'+img for img in sorted(os.listdir(image_folder)) if img.endswith(".jpg")]
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile('movie.mp4')
    return './movie.mp4'

    


  
title = "Anime2Sketch"
description = "demo for Anime2Sketch. To use it, simply upload your image, or click one of the examples to load them. Read more at the links below."
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2104.05703'>Adversarial Open Domain Adaption for Sketch-to-Photo Synthesis</a> | <a href='https://github.com/Mukosame/Anime2Sketch'>Github Repo</a></p>"

gr.Interface(
    sketch2animevid, 
    [gr.inputs.Video(type="mp4", label="Input Video") ], 
    gr.outputs.Video(label="Output"),
    title=title,
    description=description,
    article=article
   ).launch(debug=True)