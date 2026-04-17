import torch
import cv2
import numpy as np


class GradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model
        self.gradients = None
        self.activations = None

        if target_layer is None:
            self.target_layer = self.model.cv_extractor.features[0][18]
        else:
            self.target_layer = target_layer
        
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, image, input_ids, attention_mask):
        self.model.eval()
        
        with torch.enable_grad():
            image = image.requires_grad_(True)
            output = self.model(image, input_ids, attention_mask)
            
            self.model.zero_grad()
            output[0].backward()
        
        gradients = self.gradients.cpu().detach().numpy()
        activations = self.activations.cpu().detach().numpy()
        if gradients.ndim == 4:
            gradients = gradients[0]
        if activations.ndim == 4:
            activations = activations[0]
        
        weights = gradients.mean(axis=(1, 2), keepdims=True)
        cam = np.maximum(np.sum(weights * activations, axis=0), 0)
        
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cv2.resize(cam, (224, 224), interpolation=cv2.INTER_CUBIC)
    
    def overlay_heatmap(self, image, heatmap, alpha=0.4):
        if isinstance(image, torch.Tensor):
            img_np = image.cpu().squeeze().permute(1, 2, 0).numpy()
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = img_np * std + mean
            img_np = np.clip(img_np, 0, 1)
        else:
            image = image.convert("RGB")
            img_np = np.array(image).astype(np.float32) / 255.0
        
        h, w = img_np.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
        
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        overlay = (1 - alpha) * img_np + alpha * heatmap_colored / 255.0
        return np.clip(overlay, 0, 1)