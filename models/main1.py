import torch
from PIL import Image, ImageDraw
from torchvision import transforms


def preprocess_image(image_path):
    # Open the image using PIL
    image = Image.open(image_path)

    # Define the transformation pipeline
    transform = transforms.Compose([
        transforms.Resize((640, 640)),  # Resize the image to the desired input size
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
    ])

    # Apply the transformations to the image
    preprocessed_image = transform(image).unsqueeze(0)

    return preprocessed_image

def postprocess_output(output, image):
    # Extract the bounding box coordinates, class labels, and confidence scores
    pred = output[0]
    boxes = pred[:, :4].cpu().numpy()
    confidences = pred[:, 4].cpu().numpy()
    classes = pred[:, 5].cpu().numpy().astype(int)

    # Create a list to store the processed output
    processed_output = []

    # Create a copy of the image for drawing bounding boxes
    image_with_boxes = image.clone().detach()
    image_with_boxes = image_with_boxes.squeeze(0)
    image_with_boxes = transforms.ToPILImage()(image_with_boxes)
    draw = ImageDraw.Draw(image_with_boxes)

    # Iterate over the detections
    for box, conf, cls in zip(boxes, confidences, classes):
        # Extract the coordinates of the bounding box
        xmin, ymin, xmax, ymax = box
        xmin, ymin, xmax, ymax = int(xmin.item()), int(ymin.item()), int(xmax.item()), int(ymax.item())

        # Draw the bounding box on the image
        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline='red')

        # Create a dictionary to store the bounding box coordinates, label, and confidence score
        detection = {
            'label': cls,
            'score': conf,
            'bbox': [xmin, ymin, xmax, ymax]
        }

        # Add the detection to the processed output
        processed_output.append(detection)

    # Show the image with bounding boxes
    image_with_boxes.show()

    return processed_output

def main():
    # Define paths to the model weights and image
    image_path = r"C:\Users\Sanmitha\Documents\third.jpg"

    # Load the model
    model = torch.hub.load('icns-distributed-cloud/yolov5-svhn', 'svhn').fuse().eval()
    model = model.autoshape()

    # Preprocess the image
    image = preprocess_image(image_path)

    # Perform inference
    with torch.no_grad():
        outputs = model(image)

    # Postprocess the outputs and display the image with bounding boxes
    processed_output = postprocess_output(outputs, image)

    # Display or use the processed output as desired
    print(processed_output)

if __name__ == '__main__':
    main()
