import torch
import yaml
from PIL import Image
import numpy as np
from transformers import DistilBertTokenizer
import torchvision.transforms as transforms

from models.multimodal import ViralScopeModel


class InferencePipeline:
    def __init__(self, model_path, use_xai=False):
        with open('config.yaml', 'r') as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device('cpu')

        self.model = ViralScopeModel(self.config)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
        self.model.eval()

        self.tokenizer = DistilBertTokenizer.from_pretrained(
            self.config['model']['nlp']['checkpoint']
        )

        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        self.use_xai = use_xai
        if use_xai:
            try:
                from xai.gradcam import GradCAM
                from xai.integrated_gradients import TextIntegratedGradients
                self.gradcam = GradCAM(self.model)
                self.text_ig = TextIntegratedGradients(self.model, self.tokenizer)
            except Exception as e:
                print(f"XAI not available: {e}")
                self.use_xai = False

    def preprocess(self, image, title):
        if isinstance(image, Image.Image):
            image = image.convert('RGB')
            image_tensor = self.image_transform(image).unsqueeze(0)
        else:
            raise ValueError("Invalid image input")

        encoded = self.tokenizer(
            title,
            max_length=self.config['model']['nlp']['max_seq_length'] if 'max_seq_length' in self.config['model']['nlp'] else 64,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return image_tensor, encoded['input_ids'], encoded['attention_mask']

    def predict(self, image, title):
        images, input_ids, attention_mask = self.preprocess(image, title)

        images = images.to(self.device)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            prob = self.model.predict_proba(images, input_ids, attention_mask).item()

        result = {
            'probability': prob,
            'is_viral': prob > 0.5,
            'confidence': abs(prob - 0.5) * 2,
            'message': self._format_message(prob)
        }

        if self.use_xai:
            try:
                result['gradcam_heatmap'] = self.gradcam.generate(images, input_ids, attention_mask)
                result['text_attributions'] = self.text_ig.explain(title, input_ids, attention_mask, images)
            except Exception as e:
                print(f"XAI generation failed: {e}")

        return result

    def _format_message(self, prob):
        percentage = prob * 100
        if prob > 0.7:
            return f"High viral potential! {percentage:.0f}% viral score (relative to trending video distribution)."
        elif prob > 0.5:
            return f"Moderate viral potential. {percentage:.0f}% viral score (relative to trending video distribution)."
        else:
            return f"Lower viral potential. {percentage:.0f}% viral score (relative to trending video distribution)."


if __name__ == '__main__':
    import os
    model_path = 'models/best_model.pt'
    if os.path.exists(model_path):
        pipeline = InferencePipeline(model_path, use_xai=False)
        print("Pipeline initialized successfully")
    else:
        print(f"Model not found: {model_path}")