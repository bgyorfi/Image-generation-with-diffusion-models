import gradio as gr
from PIL import Image
import os

from model.generate import generate_images_with_steps
from constants import TIMESTEPS

def load_images_from_folder(folder_path):
    images = []
    if not os.path.exists(folder_path):
        return images

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path).convert("RGB")
            images.append(img)
    return images


def create_interface():
    with gr.Blocks(elem_id="container",) as app:
        initial_images = load_images_from_folder("/app/images/celeba") + load_images_from_folder("/app/images/flowers")
        with gr.Row():
            gallery = gr.Gallery(label="Generated Images", columns=4, rows=2, value=initial_images)

        with gr.Row():
            with gr.Column():
                generate_celebs = gr.Button("Generate Celebs")
            with gr.Column():
                generate_flowers = gr.Button("Generate Flowers")

        image_state = gr.State(initial_images)

        def append_flower(image_list):
            idx = 0
            for current_img, final_img in generate_images_with_steps("flowers"):
                new_gallery = [current_img] + image_list
                idx += 1
                yield gr.update(value=new_gallery), new_gallery, gr.update(value=f"Generating... {100*(idx / TIMESTEPS):.2f}%")
            yield gr.update(value=new_gallery), new_gallery, gr.update(value="Generate Flowers")

        def append_celeb(image_list):
            idx = 0
            for current_img, final_img in generate_images_with_steps("celeba"):
                new_gallery = [current_img] + image_list
                idx += 1
                yield gr.update(value=new_gallery), new_gallery, gr.update(value=f"Generating... {100*(idx / TIMESTEPS):.2f}%")
            yield gr.update(value=new_gallery), new_gallery, gr.update(value="Generate Celebs")

        generate_celebs.click(append_celeb, inputs=image_state, outputs=[gallery, image_state, generate_celebs])
        generate_flowers.click(append_flower, inputs=image_state, outputs=[gallery, image_state, generate_flowers])

    return app
