from diffusers import DiffusionPipeline
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

generator.to(device)

images = generator(
    prompt="a photo of Pikachu fine dining with a view to the Eiffel Tower",
    height=512,
    width=768,
    num_inference_steps=35,
    guidance_scale=9,
    num_images_per_prompt=1,
    negative_prompt="nude, naked",
    generator=torch.Generator().manual_seed(0)
).images

images[0].save("output35.jpg")

