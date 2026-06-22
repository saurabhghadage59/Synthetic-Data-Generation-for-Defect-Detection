from diffusers import StableDiffusionPipeline
from peft import PeftModel
import torch, os
from pathlib import Path

MODEL_ID = "runwayml/stable-diffusion-v1-5"
LORA_DIR = "models/lora_output"
GEN_DIR = Path("data/synthetic/images")

defect_prompts = {
    "color": "leather surface with color stain defect, industrial inspection, close-up photo",
    "cut":   "leather surface with cut defect, sharp edge damage, industrial inspection",
    "fold":  "leather surface with fold crease defect, industrial inspection, close-up",
    "glue":  "leather surface with glue spot contamination, industrial inspection",
    "poke":  "leather surface with poke hole puncture, industrial inspection, close-up",
}

IMAGES_PER_CLASS = 100

print("Loading SD 1.5...")
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16
)

print("Loading LoRA weights...")
pipe.unet = PeftModel.from_pretrained(pipe.unet, LORA_DIR)
pipe = pipe.to("cuda")
pipe.set_progress_bar_config(disable=True)
print("Ready! Starting generation...")

for defect, prompt in defect_prompts.items():
    out_dir = GEN_DIR / defect
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating {IMAGES_PER_CLASS} images for: {defect}")
    
    for i in range(IMAGES_PER_CLASS):
        img = pipe(
            prompt,
            negative_prompt="blurry, low quality, text, watermark, cartoon",
            num_inference_steps=25,
            guidance_scale=7.5,
            height=512, width=512
        ).images[0]
        img.save(out_dir / f"syn_{defect}_{i:04d}.png")
        
        if (i+1) % 10 == 0:
            print(f"  {i+1}/{IMAGES_PER_CLASS} done")

print("\nAll done!")
total = sum(len(list((GEN_DIR/d).glob("*.png"))) for d in defect_prompts)
print(f"Total synthetic images: {total}")