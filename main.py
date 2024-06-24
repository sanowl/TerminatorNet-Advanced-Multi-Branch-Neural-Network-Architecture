import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

class SlowNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SlowNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.sin(self.fc1(x))
        x = torch.sin(self.fc2(x))
        x = self.fc3(x)
        return x

class FastNetwork(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size):
        super(FastNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size, padding=kernel_size//2)
    
    def forward(self, x, hyper_kernel):
        batch_size, channels, height, width = x.size()
        hyper_kernel = hyper_kernel.view(batch_size, channels, 1, 1)
        x = x * hyper_kernel
        x = self.conv1(x)
        return x

class MultiBranchTerminator(nn.Module):
    def __init__(self):
        super(MultiBranchTerminator, self).__init__()
        input_channels = 3
        self.slow_net1 = SlowNetwork(input_size=2, hidden_size=256, output_size=input_channels)
        self.slow_net2 = SlowNetwork(input_size=2, hidden_size=256, output_size=input_channels)
        self.fast_net1 = FastNetwork(input_channels=input_channels, output_channels=64, kernel_size=3)
        self.fast_net2 = FastNetwork(input_channels=input_channels, output_channels=64, kernel_size=5)
        self.final_conv = nn.Conv2d(128, 3, kernel_size=1)
    
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        coordinates = torch.rand(batch_size, 2).to(x.device)
        hyper_kernel1 = self.slow_net1(coordinates)
        hyper_kernel2 = self.slow_net2(coordinates)
        
        branch1 = self.fast_net1(x, hyper_kernel1)
        branch2 = self.fast_net2(x, hyper_kernel2)
        
        combined = torch.cat((branch1, branch2), dim=1)
        output = self.final_conv(combined)
        return output

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    try:
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
        ])
        return transform(image).unsqueeze(0)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def normalize_output(tensor):
    """Normalize tensor values to be between 0 and 1"""
    min_val = tensor.min()
    max_val = tensor.max()
    return (tensor - min_val) / (max_val - min_val)

def main():
    # Initialize models
    terminator_model = MultiBranchTerminator()
    processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    hugging_face_model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k")

    # Load and preprocess image
    image_path = 'images-2.jpeg'
    image_tensor = load_and_preprocess_image(image_path)
    
    if image_tensor is None:
        return

    # Process image through TerminatorNet
    with torch.no_grad():
        terminator_output = terminator_model(image_tensor)
        
        # Normalize the output to be between 0 and 1
        terminator_output = normalize_output(terminator_output)

    # Convert to PIL Image for the processor
    to_pil = transforms.ToPILImage()
    pil_image = to_pil(terminator_output.squeeze(0))

    # Prepare input for Hugging Face model
    inputs = processor(images=pil_image, return_tensors="pt")

    # Get predictions
    with torch.no_grad():
        outputs = hugging_face_model(**inputs)
    
    logits = outputs.logits
    predicted_class = logits.argmax(-1).item()

    print(f"Predicted class: {predicted_class}")

if __name__ == "__main__":
    main()