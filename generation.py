from diffusers import DiffusionPipeline
import torch
import os

fake_path = os.path.join("images", "1")
if not os.path.exists(fake_path):
    os.makedirs(fake_path)

prompts_path = "prompts.txt"
with open(prompts_path, "r") as f:
    prompts = [line.rstrip() for line in f]

pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pipeline.to(device)

batch_size = 1
for i in range(0, len(prompts), batch_size):
    images = pipeline(prompts[i:i + batch_size], num_inference_steps=30)
    for j in range(len(images[0])):
        images[0][j].save(os.path.join(fake_path, f"{i + j}.png"))
