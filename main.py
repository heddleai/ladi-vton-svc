from vton import VTONService
from aiohttp import web
import argparse
from PIL import Image
import os
import os.path as osp

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
        async def download_to(field, pathname):
            with open(pathname, 'wb') as f:
                while True:
                    chunk = await field.read_chunk()  # 8192 bytes by default.
                    if not chunk:
                        break
                    f.write(chunk)
        # download img data from storage
        reader = await request.multipart()

        field = await reader.next()
        assert field.name == 'request_id'
        request_id = await field.read(decode=True) # will be used later on for progress
        request_id = request_id.decode()
        print("handling request", request_id)
        input_path = osp.join('inputs', request_id)
        output_path = osp.join('outputs', request_id)
        os.makedirs(input_path, exist_ok=True)
        os.makedirs(output_path, exist_ok=True)
        person_path = osp.join(input_path, 'person.jpg')
        cloth_path = osp.join(input_path, 'cloth.jpg')
        gen_path = osp.join(output_path, 'sample.jpg')

        field = await reader.next()
        assert field.name == 'person'
        await download_to(field, person_path)

        field = await reader.next()
        assert field.name == 'cloth'
        await download_to(field, cloth_path)

        person = Image.open(person_path)
        cloth = Image.open(cloth_path)
        vton.generate_image(person, cloth, gen_path)
        with open(gen_path, 'rb') as f:
            gen_img = f.read()

        os.rmdir(input_path)
        os.rmdir(output_path)

        return web.Response(body=gen_img, status=200, content_type="image/jpeg")

    app = web.Application()
    app.add_routes([
        web.get('/', status),
        web.post('/execute', handle_execute)
    ])
    web.run_app(app)

if __name__ == '__main__':
    main()