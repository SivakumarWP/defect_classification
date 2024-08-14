import gradio as gr
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification


model_path = 'model/'
num_labels = 12
test_images_dir = 'dataset/test/'
support_images_dir = 'dataset/support/'
def is_image_file(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))

test_images = sorted([file for file in os.listdir(test_images_dir) if is_image_file(file)])
support_images = sorted([file for file in os.listdir(support_images_dir) if is_image_file(file)])


image_index = -1


def load_model(model_path, num_labels):
    model = ViTForImageClassification.from_pretrained(model_path, num_labels=num_labels)
    model.eval()
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    return model


model = load_model(model_path, num_labels)


def process_image(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8'), 'RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


def predict_image(model, processed_image):
    with torch.no_grad():
        outputs = model(processed_image.to(model.device))
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)

        confidence, predicted_class = torch.max(probabilities, dim=1)
        predictions = torch.sigmoid(outputs.logits).squeeze(0)

    return predictions, confidence.item()


def update_display(test_image_path, support_image_path, target_filename, support_filename):
    test_image = Image.open(test_image_path)
    support_image = Image.open(support_image_path)
    return test_image, support_image, target_filename, support_filename

def print_class_predictions(predictions, threshold=0.5):
    class_names = [f'C{i}' for i in range(predictions.shape[0])]
    predicted_labels = [class_names[i] for i, pred in enumerate(predictions) if pred > threshold]
    return predicted_labels

def predict(source_choice, test_image, upload_image):
    if source_choice == "Upload":
        processed_image = process_image(upload_image)
    else:
        processed_image = process_image(test_image)
    predictions, confidence = predict_image(model, processed_image)
    predicted_classes = print_class_predictions(predictions)
    return f"Class: {predicted_classes}", f"Confidence: {confidence:.2%}"


def next_image():
    global image_index
    image_index = (image_index + 1) % len(test_images)
    return update_display(
        os.path.join(test_images_dir, test_images[image_index]),
        os.path.join(support_images_dir, support_images[image_index]),
        test_images[image_index],
        support_images[image_index]
    )


def previous_image():
    global image_index
    image_index = (image_index - 1) % len(test_images)
    test_image_filename = os.path.join(test_images_dir, test_images[image_index])
    support_image_filename = os.path.join(support_images_dir, support_images[image_index])

    return update_display(
        test_image_filename,
        support_image_filename,
        test_images[image_index],
        support_images[image_index]
    )


def get_initial_images():
    global image_index
    test_image_path = os.path.join(test_images_dir, test_images[image_index])
    support_image_path = os.path.join(support_images_dir, support_images[image_index])
    test_image = Image.open(test_image_path)
    support_image = Image.open(support_image_path)
    return test_image, support_image, test_images[image_index], support_images[image_index]


with gr.Blocks() as demo:
    source_selector = gr.Radio(choices=["Upload", "Test Stack"], label="Select Image Source", value="Test Stack")
    with gr.Row():
        test_image_label = gr.Textbox(label="Test Image Filename", interactive=False)
        support_image_label = gr.Textbox(label="Support Image Filename", interactive=False)
    with gr.Row():
        test_image = gr.Image(tool='editor', min_width=80, type='pil', interactive=False, label="Test Image")
        support_image = gr.Image(tool='editor', min_width=80, type='pil', interactive=False, label="Support Image")
    with gr.Row():
        gr.Button("Previous").click(previous_image, outputs=[test_image, support_image, test_image_label, support_image_label])
        gr.Button("Next").click(next_image, outputs=[test_image, support_image, test_image_label, support_image_label])
    upload_image = gr.Image(tool='editor', type='pil', interactive=True, label="Upload Image for Prediction")
    predict_button = gr.Button("Predict")
    with gr.Row():
        predicted_class = gr.Textbox(label="Predicted Class")
        prediction_score = gr.Textbox(label="Prediction Confidence")
    predict_button.click(predict, inputs=[source_selector, test_image, upload_image], outputs=[predicted_class, prediction_score])

    # Load initial images on interface start
    initial_images = get_initial_images()
    test_image.update(initial_images[0])
    support_image.update(initial_images[1])
    test_image_label.update(initial_images[2])
    support_image_label.update(initial_images[3])

demo.launch(server_name='0.0.0.0', server_port=8070, share=True)