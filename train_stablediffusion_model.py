import torch

from diffusers import StableDiffusionPipeline, StableVideoDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from safetensors import safe_open
from safetensors.torch import save_file


import os
# Set the REPLICATE_API_TOKEN environment variable
os.environ["REPLICATE_API_TOKEN"] = "r8_24nLVxrLrrVgnUHMFBqNxz7VAg9C5DG121o3S"

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def fix_diffusers_model_conversion(load_path: str, save_path: str):
    # load original
    tensors = {}
    with safe_open(load_path, framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)

    # migrate
    new_tensors = {}
    for k, v in tensors.items():
        new_key = k
        # only fix the vae
        if 'first_stage_model.' in k:
            # migrate q, k, v keys
            new_key = new_key.replace('.to_q.weight', '.q.weight')
            new_key = new_key.replace('.to_q.bias', '.q.bias')
            new_key = new_key.replace('.to_k.weight', '.k.weight')
            new_key = new_key.replace('.to_k.bias', '.k.bias')
            new_key = new_key.replace('.to_v.weight', '.v.weight')
            new_key = new_key.replace('.to_v.bias', '.v.bias')
        new_tensors[new_key] = v

    # save
    save_file(new_tensors, save_path)
    
def convert_safetensors_to_trained(model_name, save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_path = model_name
    fix_model_path = model_path.replace(".safetensors", "_fix.safetensors")  # Update the fix_model_path
    fix_diffusers_model_conversion(model_path, fix_model_path)
    
    controlnet_model_id = "lllyasviel/sd-controlnet-canny"
    
    controlnet = ControlNetModel.from_pretrained(
        controlnet_model_id, 
        torch_dtype=torch.float16
    )
    
    pipe = StableDiffusionControlNetPipeline.from_single_file(
        fix_model_path, 
        torch_dtype=torch.float16,
        controlnet=controlnet, 
        original_config_file="D:/Alfred_Workspace/Stable_Diffusion/v1-inference.yaml",
        load_safety_checker=False
    ).to(device)
    # pipe = StableDiffusionXLPipeline.from_single_file(
    #     model_path, 
    #     torch_dtype=torch.float16
    # ).to(device)
    pipe.save_pretrained(save_path)
    
    # pretrained_model = torch.load('C:/Users/XCEPT/.cache/huggingface/hub/models--naonovn--chilloutmix_NiPrunedFp32Fix')

    # # Step 2: Load the Lora SafeTensors
    # lora_safetensors = torch.load(model_name)
    
    # # Step 3: Prepare the SafeTensors for merging
    # # Assuming the SafeTensors are already in the appropriate format
    
    # # Step 4: Modify the pre-trained model if necessary
    # # Depending on the architecture of your pre-trained model, you may need to add layers or modify dimensions
    
    # # Step 5: Merge the SafeTensors with the pre-trained model
    # # Assuming the SafeTensors are compatible with the pre-trained model
    # pretrained_model.lora_safetensors = lora_safetensors
    
    # # Step 6: Fine-tune or retrain the merged model (optional)
    # # Depending on your specific use case, you may need to further train the merged model using your data
    
    # # Example usage of the merged model
    # input_data = torch.randn(1, 3, 224, 224)  # Example input data
    # output = pretrained_model(input_data)  # Forward pass through the merged model
    
    # torch.save(pretrained_model, 'merged_model.pth')
    # pipe = StableDiffusionPipeline.from_single_file(
    #     model_path
    # )
    
    # pipe = pipe.to(device)
    
    # images = pipe(
    #     prompt,
    #     width=520,
    #     height=520
    # ).images
    
    # timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    # file_name = f"generate_images/stable_diffusion_text_to_image_{timestamp}.png"
    # images[0].save(file_name)
    
model_name = "WGZ_CH.safetensors"

convert_safetensors_to_trained(
    "D:/Alfred_Workspace/Stable_Diffusion/" + model_name,
    "D:/hugging_face/model--" + model_name.replace(".safetensors", "")
)