from diffusers import StableDiffusionPipeline
import mediapy as media
import torch
import random

# Model and device setup
model_id = "dreamlike-art/dreamlike-photoreal-2.0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use "cuda" for GPU, "cpu" for CPU

# Check if model_id starts with "stabilityai/" and set model_revision accordingly
if model_id.startswith("stabilityai/"):
    model_revision = "fp16"
else:
    model_revision = None

# Create the diffusion pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float32,  # Use float32 for CPU
    revision=model_revision,
)
pipe = pipe.to(device)

# Image size setting
if model_id.endswith('-base'):
    image_width =    384  # 256  # Reduced image size for less memory usage
    image_height = 256
else:
    image_width =   768    # 384  # Reduced image size for less memory usage
    image_height = 512
# Text prompt and additional settings
prompt = "a photo of Pikachu fine dining with a view to the Eiffel Tower"
remove_safety = False
num_images = 1
seed = random.randint(0, 2147483647)

# Configure negative_prompt if remove_safety is False
if remove_safety:
    negative_prompt = None
    pipe.safety_checker = None
else:
    negative_prompt = "nude, naked"

# Generate images using the pipeline
images = pipe(
    prompt,
    height=image_height,
    width=image_width,
    num_inference_steps=15,  # Reduced number of inference steps
    guidance_scale=9,
    num_images_per_prompt=num_images,
    negative_prompt=negative_prompt,
    generator=torch.Generator().manual_seed(seed)  # Use CPU generator
).images

# Display and save images
media.show_images(images)
print(f"Seed: {seed}")
images[0].save("output.jpg")
