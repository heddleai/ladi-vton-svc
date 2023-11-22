from vton import VTONService
from aiohttp import web
import argparse
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory",
    )

    # parser.add_argument("--save_name", type=str, required=True, help="Name of the saving folder inside output_dir")
    # parser.add_argument("--test_order", type=str, required=True, choices=["unpaired", "paired"])


    # parser.add_argument("--unet_dir", required=True, type=str, help="Directory where to load the trained unet from")
    # parser.add_argument("--unet_name", type=str, default="latest",
    #                     help="Name of the unet to load from the directory specified by `--unet_dir`. "
    #                          "To load the latest checkpoint, use `latest`.")

    parser.add_argument(
        "--inversion_adapter_dir", type=str, default=None,
        help="Directory where to load the trained inversion adapter from. Required when using --text_usage=inversion_adapter",
    )
    parser.add_argument("--inversion_adapter_name", type=str, default="latest",
                        help="Name of the inversion adapter to load from the directory specified by `--inversion_adapter_dir`. "
                             "To load the latest checkpoint, use `latest`.")

    parser.add_argument("--emasc_dir", type=str, default=None,
                        help="Directory where to load the trained EMASC from. Required when --emasc_type!=none")
    parser.add_argument("--emasc_name", type=str, default="latest",
                        help="Name of the EMASC to load from the directory specified by `--emasc_dir`. "
                             "To load the latest checkpoint, use `latest`.")


    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-2-inpainting",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )


    parser.add_argument("--seed", type=int, default=1234, help="A seed for reproducible training.")

    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )

    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )

    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )


    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for the dataloader")


    parser.add_argument("--emasc_type", type=str, default='nonlinear', choices=["none", "linear", "nonlinear"],
                        help="Whether to use linear or nonlinear EMASC.")
    parser.add_argument("--emasc_kernel", type=int, default=3, help="EMASC kernel size.")
    parser.add_argument("--emasc_padding", type=int, default=1, help="EMASC padding size.")


    parser.add_argument("--text_usage", type=str, default='inversion_adapter',
                        choices=["none", "noun_chunks", "inversion_adapter"],
                        help="if 'none' do not use the text, if 'noun_chunks' use the coarse noun chunks, if "
                             "'inversion_adapter' use the features obtained trough the inversion adapter net")
    parser.add_argument("--cloth_input_type", type=str, choices=["warped", "none"], default='warped',
                        help="cloth input type. If 'warped' use the warped cloth, if none do not use the cloth as input of the unet")
    parser.add_argument("--num_vstar", default=16, type=int, help="Number of predicted v* images to use")
    parser.add_argument("--num_encoder_layers", default=1, type=int,
                        help="Number of ViT layer to use in inversion adapter")

    parser.add_argument("--use_png", default=False, action="store_true", help="Use png instead of jpg")
    parser.add_argument("--num_inference_steps", default=50, type=int, help="Number of diffusion steps")
    parser.add_argument("--guidance_scale", default=7.5, type=float, help="Guidance scale for the diffusion")
    parser.add_argument("--use_clip_cloth_features", action="store_true",
                        help="Whether to use precomputed clip cloth features")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )

    args = parser.parse_args()

    return args



def main():
    args = parse_args()
    vton = VTONService(args)

    async def status(request):
        return web.Response(text="alive")

    async def handle_execute(request):
        # download img data from storage
        person_path = "./person.jpg"
        cloth_path = "./cloth.jpg"
        person = Image.open(person_path)
        cloth = Image.open(cloth_path)
        img = vton.generate_image(person, cloth)

        # upload img data to storage
        # send reference
        ref = "some ref"
        return web.Response(text=ref)

    app = web.Application()
    app.add_routes([
        web.get('/', status),
        web.post('/execute', handle_execute)
    ])
    web.run_app(app)

if __name__ == '__main__':
    main()