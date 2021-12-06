# installations and imports
import gradio as gr
import cv2
from torchvision import transforms
import torch
import requests
import PIL
from icevision.models import mmdet
from icevision.all import *
from icevision.models.checkpoint import *
import subprocess
import sys

print("Reinstalling mmcv")
subprocess.check_call([sys.executable, "-m", "pip",
                      "uninstall", "-y", "mmcv-full==1.3.17"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "mmcv-full==1.3.17",
                      "-f", "https://download.openmmlab.com/mmcv/dist/cpu/torch1.10.0/index.html"])
print("mmcv install complete")


# class_map, metric, model parameters
classes = ['Army_navy', 'Bulldog', 'Castroviejo', 'Forceps', 'Frazier', 'Hemostat', 'Iris',
           'Mayo_metz', 'Needle', 'Potts', 'Richardson', 'Scalpel', 'Towel_clip', 'Weitlaner', 'Yankauer']
class_map = ClassMap(classes)

metrics = [COCOMetric(metric_type=COCOMetricType.bbox)]

model_type = models.mmdet.vfnet
backbone = model_type.backbones.resnet50_fpn_mstrain_2x

checkpoint_path = 'VFNet_teacher_nov29_mAP82.6.pth'
checkpoint_and_model = model_from_checkpoint(checkpoint_path)

model_loaded = checkpoint_and_model["model"]

img_size = checkpoint_and_model["img_size"]

valid_tfms = tfms.A.Adapter(
    [*tfms.A.resize_and_pad(img_size), tfms.A.Normalize()])

# gradio deployment
description1 = 'Tool for detecting 15 classes of surgical instruments:  Scalpel, Forceps, Suture needle, Clamps (Hemostat, Towel clip, Bulldog), Scissors (Mayo_metz, Iris, Potts), Needle holder (Castroviejo), Retractors (Army-navy, Richardson, Weitlaner), Suctions (Yankauer, Frazier).'
description2 = '\n \n Choose one of the examples below or use your own image of an instrument.  Click on the Submit button, allow for model prediction and see the bounding box and/or label result.'
examples = [['Image00001.jpg'], ['Image00002.jpg'], ['Image00003.jpg'], ['Image00004.jpg'], ['Image00005.jpg']]


def show_preds_gradio(input_image, display_label, display_bbox, detection_threshold):
    if detection_threshold == 0:
        detection_threshold = 0.5
    img = PIL.Image.fromarray(input_image, 'RGB')
    pred_dict = model_type.end2end_detect(img, valid_tfms, model_loaded, class_map=class_map, detection_threshold=detection_threshold,
                                          display_label=display_label, display_bbox=display_bbox, return_img=True,
                                          font_size=16, label_color="#FF59D6")
    return pred_dict['img']


display_chkbox_label = gr.inputs.Checkbox(label="Label", default=True)
display_chkbox_box = gr.inputs.Checkbox(label="Box", default=True)
detection_threshold_slider = gr.inputs.Slider(
    minimum=0, maximum=1, step=0.1, default=0.5, label="Detection Threshold")
outputs = gr.outputs.Image(type="pil")

gr_interface = gr.Interface(fn=show_preds_gradio, inputs=["image", display_chkbox_label, display_chkbox_box,  detection_threshold_slider],
                            outputs=outputs,
                            title='Surgical Instrument Detection and Identification Tool',  # , article=article,
                            description=[description1, description2],
                            examples=examples,
                            enable_queue=True)
gr_interface.launch(inline=False, share=True, debug=True)
