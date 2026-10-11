import torch


def check_env():
    print("===== AMP Environment =====")

    print("PyTorch version :", torch.__version__)
    print("CUDA available  :", torch.cuda.is_available())
    print("CUDA version    :", torch.version.cuda)

    if torch.cuda.is_available():
        print(
            "GPU             :",
            torch.cuda.get_device_name(0)
        )

        print(
            "FP16 supported  :",
            True
        )

        print(
            "BF16 supported  :",
            torch.cuda.is_bf16_supported()
        )

    else:
        print("GPU             : None")


if __name__ == "__main__":
    check_env()