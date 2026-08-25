import base64
import io
import requests
import gradio as gr
from PIL import Image

# =========================
# SGLang configuration
# =========================
SGLANG_BASE_URL = "http://127.0.0.1:30000/v1"
SGLANG_API_URL = f"{SGLANG_BASE_URL}/chat/completions"

# Keep this None so the app can automatically read the model name
# from SGLang's /v1/models endpoint.
MODEL_NAME = None

REQUEST_TIMEOUT = 300


def get_model_name():
    """Read the currently served model name from SGLang."""
    global MODEL_NAME

    if MODEL_NAME:
        return MODEL_NAME

    response = requests.get(
        f"{SGLANG_BASE_URL}/models",
        timeout=10,
    )
    response.raise_for_status()

    data = response.json()
    models = data.get("data", [])

    if not models:
        raise RuntimeError("SGLang is running, but /v1/models returned no models.")

    MODEL_NAME = models[0]["id"]
    return MODEL_NAME


def image_to_data_url(image: Image.Image) -> str:
    """Convert a PIL image to a base64 data URL."""
    image = image.convert("RGB")

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)

    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def predict(image, question):
    """
    Send one image + one question to the local SGLang OpenAI-compatible API.
    """
    if image is None:
        return "Please upload an image first."

    if not question or not question.strip():
        return "Please enter a question."

    try:
        model_name = get_model_name()
        image_url = image_to_data_url(image)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        },
                    },
                    {
                        "type": "text",
                        "text": question.strip(),
                    },
                ],
            }
        ]

        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 128,
        }

        response = requests.post(
            SGLANG_API_URL,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()
        result = response.json()

        return result["choices"][0]["message"]["content"].strip()

    except requests.exceptions.ConnectionError:
        return (
            "Unable to connect to the SGLang service.\n\n"
            "Please make sure the service is running, for example:\n"
            "python -m sglang.launch_server "
            "--model-path /root/autodl-tmp/models/"
            "llava-v1.6-vicuna-7b-scienceqa-merged "
            "--host 0.0.0.0 --port 30000"
        )

    except requests.exceptions.Timeout:
        return "SGLang request timed out. Please check whether the model is running properly or increase REQUEST_TIMEOUT."

    except requests.exceptions.HTTPError as e:
        detail = ""
        try:
            detail = response.text
        except Exception:
            pass
        return f"SGLang API request failed: {e}\n\n{detail}"

    except Exception as e:
        return f"Inference failed: {type(e).__name__}: {e}"


def check_server():
    """Check whether SGLang is alive and display the served model."""
    try:
        response = requests.get(
            f"{SGLANG_BASE_URL}/models",
            timeout=10,
        )
        response.raise_for_status()

        data = response.json()
        models = data.get("data", [])

        if not models:
            return "SGLang service is accessible, but no model was detected."

        model_names = [item.get("id", "unknown") for item in models]
        return "SGLang service is running normally.\nModels:\n" + "\n".join(model_names)

    except Exception as e:
        return f"SGLang service check failed: {type(e).__name__}: {e}"


with gr.Blocks(title="LLaVA ScienceQA") as demo:
    gr.Markdown(
        """
        # LLaVA-v1.6 ScienceQA
        **SGLang + Gradio Multimodal Visual Question Answering Demo**

        Upload a ScienceQA image, enter a question, and call the local SGLang inference service.
        """
    )

    with gr.Row():
        with gr.Column():
            image = gr.Image(
                type="pil",
                label="Image",
            )

            question = gr.Textbox(
                label="Question",
                placeholder="e.g., What is the main idea of the image?",
                lines=3,
            )

            with gr.Row():
                submit_btn = gr.Button("Submit", variant="primary")
                clear_btn = gr.ClearButton(
                    components=[image, question],
                    value="Clear",
                )

        with gr.Column():
            answer = gr.Textbox(
                label="Model Answer",
                lines=10,
                interactive=False,
            )

    with gr.Accordion("SGLang Service Status", open=False):
        status = gr.Textbox(
            label="Status",
            interactive=False,
        )
        check_btn = gr.Button("Check SGLang")

    submit_btn.click(
        fn=predict,
        inputs=[image, question],
        outputs=answer,
    )

    question.submit(
        fn=predict,
        inputs=[image, question],
        outputs=answer,
    )

    check_btn.click(
        fn=check_server,
        inputs=[],
        outputs=status,
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=6006,
        share=False,
    )