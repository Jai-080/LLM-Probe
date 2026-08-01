import os
import sys
import joblib
import torch
import numpy as np
import gradio as gr

# Include parent folder in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_probe.config import MODEL_NAME, DEVICE
from llm_probe.models.loader import load_model, load_tokenizer
from demo.inference import load_probe_payload, generate_with_probe
from demo.render import get_color_for_score

# 1. Load models globally at startup to reuse them across requests
print("Loading trained probe payload...")
payload = load_probe_payload("results/memorization/probe.joblib")
probe = payload['probe']
scaler = payload['scaler']
selected_layers = payload['selected_layers']
representation_type = payload['representation_type']

print("Loading LLM model and tokenizer...")
model = load_model(MODEL_NAME, DEVICE)
tokenizer = load_tokenizer(MODEL_NAME)

def analyze_prompt(prompt, max_tokens):
    if not prompt.strip():
        return "<div style='color: #ef4444; padding: 10px;'>Error: Prompt cannot be empty.</div>", "### Error: No prompt provided."
        
    try:
        # Run inference and score per token
        results = generate_with_probe(
            prompt=prompt,
            model=model,
            tokenizer=tokenizer,
            probe=probe,
            scaler=scaler,
            selected_layers=selected_layers,
            representation_type=representation_type,
            max_tokens=int(max_tokens)
        )
        
        # 2. Build HTML display with tooltips
        token_spans = []
        for item in results:
            token = item['token']
            score = item['mem_score']
            bg_color, border_color = get_color_for_score(score)
            
            # Format spaces and HTML entities
            escaped_token = token.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            
            # Render tokens with visual styling and hovered tooltips
            span = (
                f'<span style="background-color: {bg_color}; border-bottom: 2px solid {border_color}; '
                f'padding: 2px 4px; margin: 0 1px; border-radius: 3px; cursor: help; display: inline-block; '
                f'color: #f8fafc;" title="Score: {score:.4f}">{escaped_token}</span>'
            )
            token_spans.append(span)
            
        html_output = f'<div style="background-color: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 20px; white-space: pre-wrap; line-height: 2.2; font-size: 16px; font-family: sans-serif;">{"".join(token_spans)}</div>'
        
        # 3. Build markdown summary report
        scores = [item['mem_score'] for item in results]
        max_score = max(scores) if scores else 0.0
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        if max_score >= 0.6:
            risk_level = "HIGH"
            color_style = "color: #ef4444;"
        elif max_score >= 0.3:
            risk_level = "MEDIUM"
            color_style = "color: #eab308;"
        else:
            risk_level = "LOW"
            color_style = "color: #22c55e;"
            
        summary_md = f"""### **Response Summary**
- **Overall Memorization Risk**: <span style="font-weight: 700; {color_style}">{risk_level}</span>
- **Max Token Score**: `{max_score:.4f}`
- **Average Token Score**: `{avg_score:.4f}`
"""
        return html_output, summary_md
        
    except Exception as e:
        return f"<div style='color: #ef4444; padding: 10px;'>Generation failed: {e}</div>", f"### Generation failed\n`{e}`"

# 4. Build Custom UI layout
with gr.Blocks(title="LLM Probe - Memorization Analyzer", css="""
    .gradio-container { background-color: #0f172a; color: #f8fafc; font-family: 'Inter', sans-serif; }
    input, textarea, select { background-color: #1e293b !important; color: #f8fafc !important; border-color: #334155 !important; }
    .primary { background-color: #38bdf8 !important; color: #0f172a !important; font-weight: bold; }
    .primary:hover { background-color: #0ea5e9 !important; }
""") as demo:
    
    gr.Markdown("# 🔍 LLM Probe - Per-Token Memorization Detector")
    gr.Markdown("Autoregressively generate text and view visual, token-by-token risk scores reflecting verbatim recitation.")
    
    with gr.Row():
        with gr.Column(scale=2):
            prompt_input = gr.Textbox(
                label="Prompt", 
                placeholder="Enter a prompt to seed generation (e.g. Alice was beginning to get very tired...)", 
                lines=4
            )
            max_tokens_slider = gr.Slider(
                label="Max Generated Tokens", 
                minimum=20, 
                maximum=200, 
                value=100, 
                step=10
            )
            submit_btn = gr.Button("Analyze Generation", variant="primary")
            
        with gr.Column(scale=3):
            html_output = gr.HTML(label="Scored Generation Output")
            summary_output = gr.Markdown(label="Risk Summary")
            
    submit_btn.click(
        fn=analyze_prompt,
        inputs=[prompt_input, max_tokens_slider],
        outputs=[html_output, summary_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
