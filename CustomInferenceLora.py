import torch
from diffusers import StableDiffusionPipeline
from diffusers import PNDMScheduler

# Step 1: Load Pretrained VAE and Text Encoder
# Load a pre-trained Stable Diffusion pipeline and extract the components
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)

# Step 2: Replace the U-Net with your modified LoRA U-Net
# Assuming `apply_custom_lora_to_convs` has already been applied to `pipe.unet`
# and you have frozen the Conv2d(3x3) layers
unet_model_with_lora = pipe.unet  # This should be your modified U-Net with LoRA convolutions

# Step 3: Use a Scheduler (e.g., PNDMScheduler)
# The scheduler guides the denoising steps during inference
scheduler = PNDMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")

# Step 4: Define the Inference Function Using the Components

def run_inference(prompt, num_inference_steps=50, guidance_scale=7.5):
    """
    Run inference using the modified U-Net model with LoRA convolutions.
    
    Args:
    - prompt: The text input to condition the generation.
    - num_inference_steps: The number of denoising steps (default: 50).
    - guidance_scale: The strength of the classifier-free guidance (default: 7.5).
    
    Returns:
    - Generated image in PIL format.
    """
    # Encode the prompt using the CLIP text encoder
    text_input = pipe.tokenizer(prompt, return_tensors="pt").input_ids.cuda()

    # Get the text embeddings
    text_embeddings = pipe.text_encoder(text_input)[0].float()

    # Generate a random latent tensor to start with (noise)
    batch_size = text_embeddings.shape[0]
    latent_shape = (batch_size, 4, 64, 64)  # Assuming the latent size for U-Net is (1, 4, 64, 64)
    latents = torch.randn(latent_shape).cuda()

    # Initialize the scheduler with the correct number of timesteps
    scheduler.set_timesteps(num_inference_steps)

    # Scale the initial noise by the scheduler
    latents = latents * scheduler.init_noise_sigma

    # Step through each denoising step
    for t in scheduler.timesteps:
        # Prepare the timestep tensor
        timestep = torch.tensor([t], dtype=torch.float32).cuda()

        # Predict the noise residual using the modified U-Net with LoRA layers
        noise_pred = unet_model_with_lora(latents, timestep, encoder_hidden_states=text_embeddings)

        # Use the scheduler to compute the next latent
        latents = scheduler.step(noise_pred, t, latents)["prev_sample"]

    # Decode the latent space back into an image using the VAE
    decoded_images = pipe.vae.decode(latents / 0.18215).cpu().detach()

    # Convert the decoded image tensor into a PIL image
    images = (decoded_images / 2 + 0.5).clamp(0, 1)
    images = images.permute(0, 2, 3, 1).numpy()  # Convert to HWC format
    images = (images * 255).round().astype("uint8")

    from PIL import Image
    pil_images = [Image.fromarray(image) for image in images]
    
    return pil_images[0]  # Return the generated image (PIL format)

# Step 5: Run Inference with a Prompt
generated_image = run_inference(prompt="A futuristic cityscape at sunset", num_inference_steps=50)

# Show the generated image
generated_image.show()
