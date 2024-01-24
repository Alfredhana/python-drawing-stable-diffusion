# system packages
from io import BytesIO
import os
import time
from datetime import datetime
import threading

# pip install package
from diffusers import StableDiffusionPipeline, StableVideoDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image, export_to_video
import tkinter as tk
from tkinter import colorchooser
from PIL import ImageTk
from transformers import pipeline
import numpy as np
import torch
from PIL import Image


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import cv2
model_path = ""
lora_path = ""
width = 600
height = 600
old_x = 0
old_y = 0
brush_color = "black"
brush_size = 2

model_list = [["Painting", "model--WGZ_AB1"],
              ["Painting", "model--WGZ_AB2"],
              ["Painting", "model--WGZ_EF"],
              ["Painting", "model--WGZ_CH"],
              ["Detailed Sketch", "model--WGZ_SK"],
              ["Painting", "model--WGZ_WC"]
              ]

model_strings = [model[1] for model in model_list]

def lora_text_to_image(original_model_id, lora_model_id, positive_prompt, negative_prompt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda")
    #model_id = "runwayml/uberRealisticPornMerge_urpmv13"
    
    model_id = original_model_id
     
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    )
    
    
    lora_safetensors_path = "D:/Alfred_Workspace/Stable_Diffusion/" + lora_model_id
    
    pipe.load_lora_weights(lora_safetensors_path)
    # lora_dirs = [lora_safetensors_path]
    # lora_scales = [0.7]
    
    # for ldir, lsc in zip(lora_dirs, lora_scales):
    #     # Iteratively add new LoRA.
    #     pipe.load_lora_weights(ldir)
    #     # And scale them accordingly.
    #     pipe.fuse_lora(lora_scale = lsc)
        
    pipe = pipe.to(device)
    
    # 2. Forward embeddings and negative embeddings through text encoder
    # max_length = pipe.tokenizer.model_max_length
    
    # input_ids = pipe.tokenizer(positive_prompt, truncation=True, max_length=max_length, return_tensors="pt").input_ids
    # inputs = pipe.tokenizer.batch_encode_plus([positive_prompt], truncation=True, max_length=max_length, padding="longest", return_tensors="pt")
    # input_ids = inputs.input_ids.to(device)
    
    # negative_ids = pipe.tokenizer(negative_prompt, truncation=False, padding="max_length", max_length=input_ids.shape[-1], return_tensors="pt").input_ids                                                                                                     
    # negative_ids = negative_ids.to("cuda")
    
    # concat_embeds = []
    # neg_embeds = []
    # for i in range(0, max_length, max_length):
    #     prompt_chunk = input_ids[:, i: i + max_length]
    #     neg_chunk = negative_ids[:, i: i + max_length]
    
    #     # Apply padding if the chunk size is smaller than max_length
    #     padding_length = max_length - prompt_chunk.shape[-1]
    #     if padding_length > 0:
    #         prompt_chunk = torch.cat([prompt_chunk, torch.zeros((prompt_chunk.shape[0], padding_length), dtype=torch.long)], dim=-1)
    #         neg_chunk = torch.cat([neg_chunk, torch.zeros((neg_chunk.shape[0], padding_length), dtype=torch.long)], dim=-1)
    
    #     concat_embeds.append(pipe.text_encoder(prompt_chunk)[0])
    #     neg_embeds.append(pipe.text_encoder(neg_chunk)[0])
    
    # prompt_embeds = torch.cat(concat_embeds, dim=1)
    # negative_prompt_embeds = torch.cat(neg_embeds, dim=1)

    # images = pipe(
    #     prompt_embeds=prompt_embeds, 
    #     negative_prompt_embeds=negative_prompt_embeds,
    #     width=1280,
    #     height=1280
    # ).images
    images = pipe(
        prompt=positive_prompt, 
        negative_prompt=negative_prompt,
        width=1280,
        height=1280
    ).images
    
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = f"generate_images/stable_diffusion_text_to_image_{timestamp}.png"
    images[0].save(file_name)

def stable_diffusion_text_guided_image_to_image(prompt, image_path):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda")
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16
    )
    pipe = pipe.to(device)
    
    image = load_image(image_path)
    image = np.array(image)
    
    # get canny image
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    
    images = pipe(
        prompt=prompt, 
        image=canny_image, 
        strength=0.55, 
        guidance_scale=15,
        batch_count=2,
        batch_size=1
    ).images
    
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = f"generate_images/stable_diffusion_text_guided_image_to_image_{timestamp}.png"
    images[0].save(file_name)


def stable_diffusion_text_to_image(model_id, positive_prompt, negative_prompt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda")
    print(device)
    #model_id = "runwayml/uberRealisticPornMerge_urpmv13"
    #model_id = "digiplay/XXMix_9realistic_v1"
    #model_id = "C:/Users/XCEPT/.cache/huggingface/hub/models--naonovn--chilloutmix_NiPrunedFp32Fix"
    #model_id = "runwayml/stable-diffusion-v1-5"
    #model_id = "darkstorm2150/Protogen_x3.4_Official_Release"
    #model_id = "darkstorm2150/Protogen_x5.8_Official_Release"
    
    #model_id = "stabilityai/sd-vae-ft-mse-original"
    
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    )
    
    pipe = pipe.to(device)
    
    # 2. Forward embeddings and negative embeddings through text encoder
    # max_length = pipe.tokenizer.model_max_length
    
    # input_ids = pipe.tokenizer(positive_prompt, truncation=True, max_length=max_length, return_tensors="pt").input_ids
    # inputs = pipe.tokenizer.batch_encode_plus([positive_prompt], truncation=True, max_length=max_length, padding="longest", return_tensors="pt")
    # input_ids = inputs.input_ids.to(device)
    
    # negative_ids = pipe.tokenizer(negative_prompt, truncation=False, padding="max_length", max_length=input_ids.shape[-1], return_tensors="pt").input_ids                                                                                                     
    # negative_ids = negative_ids.to("cuda")
    
    # concat_embeds = []
    # neg_embeds = []
    # for i in range(0, max_length, max_length):
    #     prompt_chunk = input_ids[:, i: i + max_length]
    #     neg_chunk = negative_ids[:, i: i + max_length]
    
    #     # Apply padding if the chunk size is smaller than max_length
    #     padding_length = max_length - prompt_chunk.shape[-1]
    #     if padding_length > 0:
    #         prompt_chunk = torch.cat([prompt_chunk, torch.zeros((prompt_chunk.shape[0], padding_length), dtype=torch.long)], dim=-1)
    #         neg_chunk = torch.cat([neg_chunk, torch.zeros((neg_chunk.shape[0], padding_length), dtype=torch.long)], dim=-1)
    
    #     concat_embeds.append(pipe.text_encoder(prompt_chunk)[0])
    #     neg_embeds.append(pipe.text_encoder(neg_chunk)[0])
    
    # prompt_embeds = torch.cat(concat_embeds, dim=1)
    # negative_prompt_embeds = torch.cat(neg_embeds, dim=1)

    # images = pipe(
    #     prompt_embeds=prompt_embeds, 
    #     negative_prompt_embeds=negative_prompt_embeds,
    #     width=1280,
    #     height=1280
    # ).images
    images = pipe(
        prompt=positive_prompt, 
        negative_prompt=negative_prompt,
        width=600,
        height=600
    ).images
    
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = f"stable_diffusion_text_to_image_{timestamp}.png"
    image_name = "generate_images/" + file_name
    images[0].save(image_name)
    
    #upload_to_drive(file_name)

def stable_diffusion_text_guided_image_to_image(model_id, image_path, positive_prompt, negative_prompt):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda")
    model_id = "D:/hugging_face/" + model_id
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16
    )
    pipe = pipe.to(device)
    
    image = load_image(image_path)
    image = np.array(image)
    
    # get canny image
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    
    images = pipe(
        prompt=positive_prompt, 
        negative_prompt=negative_prompt,
        image=canny_image, 
        strength=0.55, 
        guidance_scale=15,
        batch_count=2,
        batch_size=1,
        width=1280,
        height=1280
    ).images
    
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = os.path.join(os.getcwd(), f"generate_images/stable_diffusion_text_guided_image_to_image_{timestamp}.png")
    images[0].save(file_name)

def stable_diffusion_image_to_video(image_path):
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt", 
        torch_dtype=torch.float16, variant="fp16"
    )
    
    pipe.enable_model_cpu_offload()
    
    # Load the conditioning image
    image = load_image(image_path)
    image = image.resize((1024, 576))
    
    generator = torch.manual_seed(42)
    frames = pipe(image, decode_chunk_size=8, generator=generator).frames[0]
    
    export_to_video(frames, "generated.mp4", fps=7)

   
def image_to_text(image_path):
    captioner = pipeline(
        "image-to-text",
        model="Salesforce/blip-image-captioning-base"
    )
    text = captioner(image_path)
    
    return text

class DrawStretchCommand:
    def __init__(self, canvas, x1, y1, x2, y2, brush_size, brush_color):
        self.canvas = canvas
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.brush_size = brush_size
        self.brush_color = brush_color
        self.line_id = None
    
    def execute(self):
        self.line_id = self.canvas.create_line(
            self.x1, self.y1, self.x2, self.y2, width=self.brush_size, fill=self.brush_color
        )
    
    def undo(self):
        self.canvas.delete(self.line_id)

class CanvasApp:
    def __init__(self, width, height):
        self.root = tk.Tk()
        self.root.title("Stretch Drawing Program")
        self.width = self.root.winfo_screenwidth()
        frame_height = 120
        self.height = self.root.winfo_screenheight() - frame_height
        self.root.attributes("-fullscreen", True)
        self.root.geometry(f"{width}x{height}")

        self.canvas = tk.Canvas(self.root, width=self.width - frame_height, height=self.height, highlightthickness=1, highlightbackground="black", background='white')
        self.canvas.pack(side=tk.TOP, pady=20)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.start_stretch)
        self.canvas.bind("<B1-Motion>", self.draw_stretch)
        self.canvas.bind("<ButtonRelease-1>", self.end_stretch)
        self.undo_stack = []

        button_size = 5
        button_frame = tk.Frame(self.root, width=width, height=frame_height)
        button_frame.pack(side=tk.BOTTOM, pady=5)
        
        button_padding = 10

        # paint_button = tk.Button(button_frame, text="Paint", command=self.paint_canvas)
        # paint_button.pack(side=tk.LEFT, padx=button_padding)
        # paint_button.config(height=button_size, width=button_size * 2)
        # clear_button = tk.Button(button_frame, text="Clear", command=self.clear_canvas)
        # clear_button.pack(side=tk.LEFT, padx=button_padding)
        # clear_button.config(height=button_size, width=button_size * 2)
        undo_button = tk.Button(button_frame, text="Undo", command=self.undo_stretch)
        undo_button.pack(side=tk.LEFT, padx=button_padding)
        undo_button.config(height=button_size, width=button_size * 2)
        reset_button = tk.Button(button_frame, text="Reset", command=self.reset_canvas)
        reset_button.pack(side=tk.LEFT, padx=button_padding)
        reset_button.config(height=button_size, width=button_size * 2)
        save_button = tk.Button(button_frame, text="Save", command=self.save_canvas)
        save_button.pack(side=tk.LEFT, padx=button_padding)
        save_button.config(height=button_size, width=button_size * 2)
        color_button = tk.Button(button_frame, text="Change Color", command=self.change_color)
        color_button.pack(side=tk.LEFT, padx=button_padding)
        color_button.config(height=button_size, width=button_size * 2)
        size_var = tk.StringVar()
        size_var.set("2")
        size_label = tk.Label(button_frame, text="Brush Size:")
        size_label.pack(side=tk.LEFT)
        size_dropdown = tk.OptionMenu(button_frame, size_var, *range(21))
        size_dropdown.config(height=button_size, width=button_size)
        size_dropdown.pack(side=tk.LEFT, padx=button_padding)
        size_var.trace("w", self.change_size)
        
        self.model_string = model_strings[0]
        model_var = tk.StringVar()
        model_var.set(self.model_string)
        model_label = tk.Label(button_frame, text="Model Number:")
        model_label.pack(side=tk.LEFT)
        model_dropdown = tk.OptionMenu(button_frame, model_var, *model_strings, command=self.change_model)
        model_dropdown.config(height=button_size, width=button_size * 4)
        model_dropdown.pack(side=tk.LEFT, padx=button_padding)
        
        self.brush_size = int(size_var.get())
        self.brush_color = "black"
        self.clear_color = "white"
        self.paint_color = self.brush_color

        self.is_drawing = False
        self.current_stretch = None
        # Create a loading window
        self.loading_window = None
        self.generated_image_label = None
        
    def control_net_text_guided_image_to_image(self, model_id, image_path, positive_prompt, negative_prompt, weight: float, progress_callback=None):
        #controlnet_model_id = "D:/HKMOA/models--sd-controlnet-canny"
        controlnet_model_id = "lllyasviel/sd-controlnet-canny"
        model_id = "D:/HKMOA/" + model_id
        image = load_image(image_path)
        image = np.array(image)
        
        # get canny image
        image = cv2.Canny(image, 100, 200)
        image = image[:, :,
                      None]
        
        image = np.concatenate([image, image, image], axis=2)
        canny_image = Image.fromarray(image)
        
        controlnet = ControlNetModel.from_pretrained(
            controlnet_model_id, 
            torch_dtype=torch.float16
        )
        
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            model_id, 
            controlnet=controlnet, 
            torch_dtype=torch.float16
        )
        
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_model_cpu_offload()
        
        generator = torch.manual_seed(0)
        
        self.total_steps = 35  # Total number of inference steps
        
        def callback_dynamic_cfg(pipe, step_index, timestep, callback_kwargs):
            print(f"step: {step_index}, total_steps: {self.total_steps}")
            if progress_callback is not None:
                progress_callback(int(step_index), self.total_steps)
            return callback_kwargs
        
        images = pipe(
            prompt=positive_prompt, 
            negative_prompt=negative_prompt,
            num_inference_steps=self.total_steps, 
            generator=generator, 
            image=canny_image,
            controlnet_conditioning_scale = weight,
            control_guidance_start = 0,
            control_guidance_end = 1,
            callback_on_step_end=callback_dynamic_cfg,
        ).images
        
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        file_name = os.path.join(os.getcwd(), f"generate_images/control_weight_{weight}_{timestamp}.png")
        images[0].save(file_name)
        
        return file_name
        
    def show_loading_window(self):
        self.loading_window = tk.Toplevel()
        self.loading_window.title("Loading")
        self.loading_window.geometry("200x100")
        self.loading_label = tk.Label(self.loading_window, text="Loading...")
        self.loading_label.pack()
        
        self.progress_label = tk.Label(self.loading_window)
        self.progress_label.pack()
        
        self.progress_label.config(text="Inference Progress:")
        
        root = self.canvas.winfo_toplevel()  # Access the root window object
        root.update_idletasks()  # Update the root window # Force the main window to update and display the loading window
        
        
    def update_progress(self, step, total_steps):
        
        self.progress_label.config(text=f"Inference Progress: {step} / {total_steps}")
        
        root = self.loading_window  # Access the root window object
        root.update_idletasks()  # Update the root window # Force the main window to update and display the loading window
        
    def save_canvas(self):
        global model_list
        
        # Show the loading window
        self.show_loading_window()
        root = self.canvas.winfo_toplevel()  # Access the root window object
        root.update_idletasks()  # Update the root window # Force the main window to update and display the loading window
        
        thread = threading.Thread(target=self.process_image)
        thread.daemon = True  # Set the thread as a daemon to automatically exit when the main program ends
        thread.start()
    
    def process_image(self):
        try:
            print("Saving canvas...")
            
            # Get the image data from the canvas
            image_data = self.canvas.postscript(colormode="color")
    
            # Convert the postscript data to image format
            image = Image.open(BytesIO(image_data.encode("utf-8")))
            image = image.convert("RGBA")
            
            print("Image processing....")
            start_time = time.time()
            
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            image_name = os.path.join(os.getcwd(), "generate_images/temp_file_" + f"{timestamp}.png")
            
            # Save the image to a file
            image.save(image_name)
            
            # model = model_list[self.model_index]
            # model_type = model[0]
            # model_name = model[1]
            
            model_type = model_list[model_strings.index(self.model_string)][0]
            model_name = self.model_string
            
            prompt_text = "a {0} in the style of {1}".format(model_type, model_name)
    
            print("Generated Text:", prompt_text)
            
            print("Canvas saved successfully.")
        
            generate_file_name = self.control_net_text_guided_image_to_image(
                model_name, 
                image_name, 
                prompt_text, 
                "", 
                1.0, 
                self.update_progress
            )
            
            # Get the end time
            end_time = time.time()
            
            # Calculate the duration
            duration = end_time - start_time
            print("Finished....")
            print("Time taken: %.2f seconds" % duration)
            
            # Show the generated image in the loading window
            self.show_generated_image(generate_file_name)
            
        except Exception as e:
            print("An error occurred while saving the canvas:")
            print(str(e))
            
            # Close the loading window in case of an error
            #self.close_loading_window()
    
    def show_generated_image(self, image_path):
        # Load the generated image using PIL
        image = Image.open(image_path)
        
        self.loading_label.config(text="")
        
        # Get the size of the generated image
        image_width, image_height = image.size
        self.progress_label.config(text="Inference Progress: Finished")
        
        # Resize the loading window to fit the image
        self.loading_window.geometry(f"{image_width + 20}x{image_height + 40}")
        
        # Convert the PIL image to Tkinter-compatible image
        image_tk = ImageTk.PhotoImage(image)
        
        # Create a label to display the generated image
        self.generated_image_label = tk.Label(self.loading_window, image=image_tk)
        # Keep a reference to the image to prevent garbage collection
        self.generated_image_label.image = image_tk
        self.generated_image_label.pack()
    
    def paint_canvas(self):
        self.paint_color = self.brush_color
        self.brush_size = int(2)
        
    def clear_canvas(self):
        self.paint_color = self.clear_color
        self.brush_size = int(30)
        
    def start_stretch(self, event):
        self.is_drawing = True
        self.current_stretch = []

    def draw_stretch(self, event):
        if self.is_drawing:
            if len(self.current_stretch) >= 2:
                self.canvas.create_line(
                    self.current_stretch[-1][0], self.current_stretch[-1][1],
                    event.x, event.y, width=self.brush_size, fill=self.paint_color
                )
            self.current_stretch.append((event.x, event.y))

    def end_stretch(self, event):
        if self.is_drawing:
            self.current_stretch.append((event.x, event.y))
            self.undo_stack.append(list(self.current_stretch))
            self.is_drawing = False

    def change_color(self):
        color = colorchooser.askcolor(title="Select Brush Color")
        if color:
            self.brush_color = color[1]

    def change_size(self, *args):
        self.brush_size = int(args[0])
        
    def change_model(self, args):
        self.model_string = args

    def reset_canvas(self):
        self.canvas.delete("all")
        self.undo_stack = []

    def undo_stretch(self):
        if self.undo_stack:
            last_stretch = self.undo_stack.pop()
            self.canvas.delete("all")
            for i in range(len(self.undo_stack)):
                stretch = self.undo_stack[i]
                for j in range(1, len(stretch)):
                    self.canvas.create_line(
                        stretch[j-1][0], stretch[j-1][1],
                        stretch[j][0], stretch[j][1],
                        width=self.brush_size, fill=self.brush_color
                    )
            
    def run(self):
        self.root.mainloop()


# Create and run the fullscreen canvas app
app = CanvasApp(width, height)
app.run()
#stable_diffusion_image_to_video("images/girl.png")

# image_to_text("images\ship.jpg")
                                   
# stable_diffusion_text_guided_image_to_image(
#     "A boat floating on top of water, wu guan zhong style", "images\ship.jpg"
# )

# control_net_text_guided_image_to_image(
#     "A boat floating on top of water, wu guan zhong style", "images\ship.jpg"
# )