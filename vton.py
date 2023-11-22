import os
import torch
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import UNet2DConditionModel, DDIMScheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, AutoProcessor
from img_preprocessor import ImgPreprocessor

from ladi_vton.src.models.AutoencoderKL import AutoencoderKL
from ladi_vton.src.models.emasc import EMASC
from ladi_vton.src.models.inversion_adapter import InversionAdapter
from ladi_vton.src.utils.image_from_pipe import generate_images_from_tryon_pipe
from ladi_vton.src.utils.set_seeds import set_seed
from ladi_vton.src.vto_pipelines.tryon_pipe import StableDiffusionTryOnePipeline

from transformers import CLIPVisionModelWithProjection, CLIPProcessor

import torchvision
from ladi_vton.src.utils.encode_text_word_embedding import encode_text_word_embedding

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__, log_level="INFO")
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["WANDB_START_METHOD"] = "thread"


@torch.no_grad()
def generate_images_from_tryon_pipe(pipe: StableDiffusionTryOnePipeline, inversion_adapter: InversionAdapter,
                                    batch: dict, output_dir: str, order: str,
                                    save_name: str, text_usage: str, vision_encoder: CLIPVisionModelWithProjection,
                                    processor: CLIPProcessor, cloth_input_type: str, cloth_cond_rate: int = 1,
                                    num_vstar: int = 1, seed: int = 1234, num_inference_steps: int = 50,
                                    guidance_scale: int = 7.5, use_png: bool = False):
    # Create output directory
    save_path = os.path.join(output_dir, f"{save_name}_{order}")
    os.makedirs(save_path, exist_ok=True)

    # Set seed
    generator = torch.Generator("cuda").manual_seed(seed)
    num_samples = 1

    # Generate images
    model_img = batch.get("image")
    mask_img = batch.get("inpaint_mask")
    if mask_img is not None:
        mask_img = mask_img.type(torch.float32)
    pose_map = batch.get("pose_map")
    warped_cloth = batch.get('warped_cloth')
    category = batch.get("category")

    # Generate text prompts
    if text_usage == "noun_chunks":
        prompts = batch["captions"]
    elif text_usage == "none":
        prompts = [""] * len(batch["captions"])
    elif text_usage == 'inversion_adapter':
        category_text = {
            'dresses': 'a dress',
            'upper_body': 'an upper body garment',
            'lower_body': 'a lower body garment',

        }
        text = [f'a photo of a model wearing {category_text[category]} {" $ " * num_vstar}' for
                category in batch['category']]

        clip_cloth_features = batch.get('clip_cloth_features')
        if clip_cloth_features is None:
            with torch.no_grad():
                # Get the visual features of the in-shop cloths
                input_image = torchvision.transforms.functional.resize((batch["cloth"] + 1) / 2, (224, 224),
                                                                        antialias=True).clamp(0, 1)
                processed_images = processor(images=input_image, return_tensors="pt")
                clip_cloth_features = vision_encoder(
                    processed_images.pixel_values.to(model_img.device)).last_hidden_state

        # Compute the predicted PTEs
        word_embeddings = inversion_adapter(clip_cloth_features.to(model_img.device))
        word_embeddings = word_embeddings.reshape((word_embeddings.shape[0], num_vstar, -1))

        # Tokenize text
        tokenized_text = pipe.tokenizer(text, max_length=pipe.tokenizer.model_max_length, padding="max_length",
                                        truncation=True, return_tensors="pt").input_ids
        tokenized_text = tokenized_text.to(word_embeddings.device)

        # Encode the text using the PTEs extracted from the in-shop cloths
        encoder_hidden_states = encode_text_word_embedding(pipe.text_encoder, tokenized_text,
                                                            word_embeddings, num_vstar).last_hidden_state
    else:
        raise ValueError(f"Unknown text usage {text_usage}")

    # Generate images
    if text_usage == 'inversion_adapter':
        generated_images = pipe(
            image=model_img,
            mask_image=mask_img,
            pose_map=pose_map,
            warped_cloth=warped_cloth,
            prompt_embeds=encoder_hidden_states,
            height=512,
            width=384,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_samples,
            generator=generator,
            cloth_input_type=cloth_input_type,
            cloth_cond_rate=cloth_cond_rate,
            num_inference_steps=num_inference_steps
        ).images
    else:
        generated_images = pipe(
            prompt=prompts,
            image=model_img,
            mask_image=mask_img,
            pose_map=pose_map,
            warped_cloth=warped_cloth,
            height=512,
            width=384,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_samples,
            generator=generator,
            cloth_input_type=cloth_input_type,
            cloth_cond_rate=cloth_cond_rate,
            num_inference_steps=num_inference_steps
        ).images

    # Save images
    for gen_image, cat, name in zip(generated_images, category, batch["im_name"]):
        if not os.path.exists(os.path.join(save_path, cat)):
            os.makedirs(os.path.join(save_path, cat))

        if use_png:
            name = name.replace(".jpg", ".png")
            gen_image.save(
                os.path.join(save_path, cat, name))
        else:
            gen_image.save(
                os.path.join(save_path, cat, name), quality=95)


@torch.inference_mode()
class VTONService():
    def __init__(self, args):
        self.args = args
        # Enable TF32 for faster inference on Ampere GPUs,
        # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        if args.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        # Setup accelerator and device.
        accelerator = Accelerator()
        device = accelerator.device

        # If passed along, set the training seed now.
        if args.seed is not None:
            set_seed(args.seed)

        # Load scheduler, tokenizer and models.
        val_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        val_scheduler.set_timesteps(args.num_inference_steps, device=device)
        text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
        vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
        tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")

        # Define the extended unet
        new_in_channels = 27 if args.cloth_input_type == "none" else 31
        # the posemap has 18 channels, the (encoded) cloth has 4 channels, the standard SD inpaining has 9 channels
        with torch.no_grad():
            # Replace the first conv layer of the unet with a new one with the correct number of input channels
            conv_new = torch.nn.Conv2d(
                in_channels=new_in_channels,
                out_channels=unet.conv_in.out_channels,
                kernel_size=3,
                padding=1,
            )

            torch.nn.init.kaiming_normal_(conv_new.weight)  # Initialize new conv layer
            conv_new.weight.data = conv_new.weight.data * 0.  # Zero-initialize new conv layer

            conv_new.weight.data[:, :9] = unet.conv_in.weight.data  # Copy weights from old conv layer
            conv_new.bias.data = unet.conv_in.bias.data  # Copy bias from old conv layer

            unet.conv_in = conv_new  # replace conv layer in unet
            unet.config['in_channels'] = new_in_channels  # update config

        # Load the trained unet
        if args.unet_name != "latest":
            path = args.unet_name
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.unet_dir)
            dirs = [d for d in dirs if d.startswith("unet")]
            dirs = sorted(dirs, key=lambda x: int(os.path.splitext(x.split("_")[-1])[0]))
            path = dirs[-1]
        accelerator.print(f"Loading Unet checkpoint {path}")
        unet.load_state_dict(torch.load(os.path.join(args.unet_dir, path)))

        if args.emasc_type != 'none':
            # Define EMASC model.
            in_feature_channels = [128, 128, 128, 256, 512]
            out_feature_channels = [128, 256, 512, 512, 512]
            int_layers = [1, 2, 3, 4, 5]

            emasc = EMASC(in_feature_channels,
                          out_feature_channels,
                          kernel_size=args.emasc_kernel,
                          padding=args.emasc_padding,
                          stride=1,
                          type=args.emasc_type)

            if args.emasc_dir is not None:
                if args.emasc_name != "latest":
                    path = args.emasc_name
                else:
                    # Get the most recent checkpoint
                    dirs = os.listdir(args.emasc_dir)
                    dirs = [d for d in dirs if d.startswith("emasc")]
                    dirs = sorted(dirs, key=lambda x: int(os.path.splitext(x.split("_")[-1])[0]))
                    path = dirs[-1]
                accelerator.print(f"Loading EMASC checkpoint {path}")
                emasc.load_state_dict(torch.load(os.path.join(args.emasc_dir, path)))
            else:
                raise ValueError("No EMASC checkpoint found. Make sure to specify --emasc_dir")
        else:
            emasc = None
            int_layers = None

        if args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")

        # add posemap input to unet
        outputlist = ['image', 'pose_map', 'captions', 'inpaint_mask', 'im_mask', 'category', 'im_name']

        if args.cloth_input_type == 'warped':
            outputlist.append('warped_cloth')

        if args.text_usage == 'inversion_adapter':
            if args.pretrained_model_name_or_path == "runwayml/stable-diffusion-inpainting":
                vision_encoder = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
                processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
            elif args.pretrained_model_name_or_path == "stabilityai/stable-diffusion-2-inpainting":
                vision_encoder = CLIPVisionModelWithProjection.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
                processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
            else:
                raise ValueError(f"Unknown pretrained model name or path: {args.pretrained_model_name_or_path}")
            vision_encoder.requires_grad_(False)

            inversion_adapter = InversionAdapter(input_dim=vision_encoder.config.hidden_size,
                                                hidden_dim=vision_encoder.config.hidden_size * 4,
                                                output_dim=text_encoder.config.hidden_size * args.num_vstar,
                                                num_encoder_layers=args.num_encoder_layers,
                                                config=vision_encoder.config)

            if args.inversion_adapter_dir is not None:
                if args.inversion_adapter_name != "latest":
                    path = args.inversion_adapter_name
                else:
                    # Get the most recent checkpoint
                    dirs = os.listdir(args.inversion_adapter_dir)
                    dirs = [d for d in dirs if d.startswith("inversion_adapter")]
                    dirs = sorted(dirs, key=lambda x: int(os.path.splitext(x.split("_")[-1])[0]))
                    path = dirs[-1]
                accelerator.print(f"Loading inversion adapter checkpoint {path}")
                inversion_adapter.load_state_dict(torch.load(os.path.join(args.inversion_adapter_dir, path)))
            else:
                raise ValueError("No inversion adapter checkpoint found. Make sure to specify --inversion_adapter_dir")

            inversion_adapter.requires_grad_(False)

            if args.use_clip_cloth_features:
                outputlist.append('clip_cloth_features')
                vision_encoder = None
            else:
                outputlist.append('cloth')
        else:
            inversion_adapter = None
            vision_encoder = None
            processor = None


        weight_dtype = torch.float16

        # Move models to device and eval mode
        text_encoder.to(device, dtype=weight_dtype)
        text_encoder.eval()
        vae.to(device, dtype=weight_dtype)
        vae.eval()
        unet.to(device, dtype=weight_dtype)
        unet.eval()
        if emasc is not None:
            emasc.to(device, dtype=weight_dtype)
            emasc.eval()
        if inversion_adapter is not None:
            inversion_adapter.to(device, dtype=weight_dtype)
            inversion_adapter.eval()
        if vision_encoder is not None:
            vision_encoder.to(device, dtype=weight_dtype)
            vision_encoder.eval()

        self.inversion_adapter = inversion_adapter
        self.vision_encoder = vision_encoder
        self.processor = processor
        # Define the pipeline
        self.val_pipe = StableDiffusionTryOnePipeline(
            text_encoder=text_encoder,
            vae=vae,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=val_scheduler,
            emasc=emasc,
            emasc_int_layers=int_layers,
        ).to(device)
        self.img_preprocessor = ImgPreprocessor()

    def generate_image(self, person_image, cloth_image):
        batch = self.img_preprocessor.preprocess(person_image, cloth_image)
        # Generate images
        with torch.cuda.amp.autocast():
            generate_images_from_tryon_pipe(self.val_pipe, self.inversion_adapter, batch, self.args.output_dir,
                                            self.args.test_order, self.args.save_name, self.args.text_usage, self.vision_encoder, self.processor,
                                            self.args.cloth_input_type, 1, self.args.num_vstar, self.args.seed,
                                            self.args.num_inference_steps, self.args.guidance_scale, self.args.use_png)

