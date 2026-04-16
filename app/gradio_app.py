import gradio as gr
from PIL import Image
import numpy as np

from app.inference import InferencePipeline

pipeline = InferencePipeline(
    model_path='models/best_model.pt',
    use_xai=True
)

def analyze_video(thumbnail, title):
    if thumbnail is None:
        return "Please upload a thumbnail image.", None, None
    
    if not title or title.strip() == "":
        return "Please enter a video title.", None, None
    
    result = pipeline.predict(thumbnail, title.strip())
    
    message = result['message']
    message += f"\n\nProbability: {result['probability']*100:.1f}%"
    message += f"\nConfidence: {result['confidence']*100:.1f}%"
    
    gradcam_image = None
    title_html = None
    
    if 'gradcam_heatmap' in result:
        heatmap = result['gradcam_heatmap']
        gradcam_image = pipeline.gradcam.overlay_heatmap(thumbnail, heatmap)
        gradcam_image = (gradcam_image * 255).astype(np.uint8)
    
    if 'text_attributions' in result:
        title_html = pipeline.text_ig.format_explanation(result['text_attributions'])
        title_html = f"<div style='font-size: 18px; padding: 10px;'>{title_html}</div>"
    
    return message, gradcam_image, title_html

with gr.Blocks(title=pipeline.config['deployment']['gradio']['title']) as demo:
    gr.Markdown(f"# {pipeline.config['deployment']['gradio']['title']}")
    gr.Markdown(pipeline.config['deployment']['gradio']['description'])
    
    with gr.Row():
        with gr.Column():
            thumbnail_input = gr.Image(
                type="pil",
                label=pipeline.config['deployment']['gradio']['thumbnail_label']
            )
            title_input = gr.Textbox(
                label=pipeline.config['deployment']['gradio']['title_label'],
                placeholder="Enter your video title..."
            )
            submit_btn = gr.Button(
                pipeline.config['deployment']['gradio']['submit_button'],
                variant="primary"
            )
        
        with gr.Column():
            output_message = gr.Textbox(label="Prediction", lines=3)
            gradcam_output = gr.Image(label="Visual Attention Heatmap")
            title_output = gr.HTML(label="Title Word Impact")
    
    submit_btn.click(
        fn=analyze_video,
        inputs=[thumbnail_input, title_input],
        outputs=[output_message, gradcam_output, title_output]
    )

if __name__ == '__main__':
    demo.launch()