# import argparse
# from llava.model.builder import load_pretrained_model
# from llava.mm_utils import get_model_name_from_path


# def merge_lora(args):
#     model_name = get_model_name_from_path(args.model_path)
#     tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, device_map='cpu')

#     model.save_pretrained(args.save_model_path)
#     tokenizer.save_pretrained(args.save_model_path)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model-path", type=str, required=True)
#     parser.add_argument("--model-base", type=str, required=True)
#     parser.add_argument("--save-model-path", type=str, required=True)

#     args = parser.parse_args()

#     merge_lora(args)



# import argparse

# from llava.model.builder import load_pretrained_model


# def merge_lora(args):
#     # IMPORTANT:
#     # builder.py uses "llava" and "lora" in model_name
#     # to select the LLaVA LoRA loading branch.
#     model_name = "llava-v1.6-vicuna-7b-lora"

#     tokenizer, model, image_processor, context_len = load_pretrained_model(
#         args.model_path,
#         args.model_base,
#         model_name,
#         device_map="cpu",
#     )

#     model.save_pretrained(args.save_model_path)
#     tokenizer.save_pretrained(args.save_model_path)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()

#     parser.add_argument("--model-path", type=str, required=True)
#     parser.add_argument("--model-base", type=str, required=True)
#     parser.add_argument("--save-model-path", type=str, required=True)

#     args = parser.parse_args()
#     merge_lora(args)




import argparse

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path


def merge_lora(args):

    # 自动从 LoRA 输出目录获取模型名称
    model_name = get_model_name_from_path(args.model_path)

    print("LoRA path:", args.model_path)
    print("Base model:", args.model_base)
    print("Model name:", model_name)
    print("Save path:", args.save_model_path)

    # CPU 加载 Base + LoRA，并完成 merge
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        device_map="cpu"
    )

    print("LoRA loaded and merged.")

    # 保存合并后的完整模型
    model.save_pretrained(args.save_model_path)
    tokenizer.save_pretrained(args.save_model_path)

    print("Merged model saved to:", args.save_model_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to LoRA adapter"
    )

    parser.add_argument(
        "--model-base",
        type=str,
        required=True,
        help="Path to base LLaVA model"
    )

    parser.add_argument(
        "--save-model-path",
        type=str,
        required=True,
        help="Path to save merged model"
    )

    args = parser.parse_args()

    merge_lora(args)
