import torch
import numpy as np
from captum.attr import IntegratedGradients


class TextIntegratedGradients:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.ig = IntegratedGradients(self._forward_embeddings)

    def _forward_embeddings(self, inputs_embeds, attention_mask, cv_features):
        outputs = self.model.nlp_extractor.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        )
        cls_features = outputs.last_hidden_state[:, 0, :]
        logits = self.model.fusion(cv_features, cls_features)
        return logits

    def explain(self, text, input_ids, attention_mask, image):
        self.model.eval()
        
        with torch.no_grad():
            input_embeddings = self.model.nlp_extractor.backbone.embeddings(
                input_ids=input_ids
            )
            pad_token_id = self.tokenizer.pad_token_id
            baseline_input_ids = torch.full_like(input_ids, pad_token_id)
            baseline_embeddings = self.model.nlp_extractor.backbone.embeddings(
                input_ids=baseline_input_ids
            )
            cv_features = self.model.cv_extractor(image)
        
        input_embeddings = input_embeddings.requires_grad_(True)
        
        attributions, delta = self.ig.attribute(
            input_embeddings,
            baselines=baseline_embeddings,
            additional_forward_args=(attention_mask, cv_features),
            target=None,
            return_convergence_delta=True,
            n_steps=50
        )
        
        token_attributions = attributions.sum(dim=-1).squeeze().detach().cpu().numpy()
        
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
        
        results = []
        for token, score in zip(tokens, token_attributions):
            if token not in ['[PAD]', '[CLS]', '[SEP]']:
                results.append((token, float(score)))
        
        return results

    def format_explanation(self, attributions):
        formatted = []
        for token, score in attributions:
            display_token = token.replace('##', '')
            
            if score > 0:
                intensity = min(abs(score) * 5, 1.0)
                color = f'rgba(0, 255, 0, {intensity})'
            else:
                intensity = min(abs(score) * 5, 1.0)
                color = f'rgba(255, 0, 0, {intensity})'
            
            formatted.append(f'<span style="background-color: {color}">{display_token}</span>')
        
        return ' '.join(formatted)