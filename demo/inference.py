import os
import sys
import argparse
import joblib
import torch
import numpy as np

# Include parent folder in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_probe.config import MODEL_NAME, DEVICE
from llm_probe.models.loader import load_model, load_tokenizer

def load_probe_payload(probe_path="results/memorization/probe.joblib"):
    """
    Loads the saved probe and scaler payload.
    """
    if not os.path.exists(probe_path):
        raise FileNotFoundError(f"Trained probe payload not found at: {probe_path}")
    payload = joblib.load(probe_path)
    return payload

def generate_with_probe(prompt, model, tokenizer, probe, scaler, selected_layers, representation_type, max_tokens=100):
    """
    Generates text token-by-token and predicts the memorization score for each new token.
    Uses a growing-window mean-pool approximation for 'mean_pooled' representation.
    """
    # 1. Encode prompt
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_len = len(prompt_ids)
    
    device = next(model.parameters()).device
    
    # 2. Initial forward pass to predict the first token
    input_ids_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask_tensor = torch.ones_like(input_ids_tensor, device=device)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids_tensor,
            attention_mask=attention_mask_tensor,
            output_hidden_states=True
        )
        
    next_token_logits = outputs.logits[0, -1, :]
    next_token_id = torch.argmax(next_token_logits).item()
    
    gen_ids = []
    results = []
    
    # 3. Autoregressive loop
    while len(gen_ids) < max_tokens:
        if next_token_id == tokenizer.eos_token_id:
            break
            
        gen_ids.append(next_token_id)
        token_str = tokenizer.decode([next_token_id])
        
        # Forward pass on the updated sequence (including the new token)
        combined_ids = prompt_ids + gen_ids
        input_ids_tensor = torch.tensor([combined_ids], dtype=torch.long, device=device)
        attention_mask_tensor = torch.ones_like(input_ids_tensor, device=device)
        
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids_tensor,
                attention_mask=attention_mask_tensor,
                output_hidden_states=True
            )
            
        hidden_states = outputs.hidden_states
        
        # 4. Extract layer representations for the new prefix continuation
        example_feat = []
        for layer in selected_layers:
            # Shape: (seq_len, hidden_dim)
            layer_hidden_seq = hidden_states[layer][0]
            # Slice continuation part: shape (continuation_len, hidden_dim)
            continuation_hidden = layer_hidden_seq[prompt_len:, :]
            
            if representation_type == 'last_token':
                # Grab activation of the final generated token
                feat_vec = continuation_hidden[-1, :]
            else:
                # mean_pooled over the growing window of generated tokens
                feat_vec = continuation_hidden.mean(dim=0)
                
            example_feat.append(feat_vec.cpu().to(torch.float32).numpy())
            
        # Concatenate in the exact order selected_layers was stored
        X_token = np.concatenate(example_feat).reshape(1, -1)
        
        # Standardize using the loaded training scaler
        X_token_scaled = scaler.transform(X_token)
        
        # Predict probability of class 1 (memorized)
        mem_score = probe.predict_proba(X_token_scaled)[0, 1]
        
        results.append({
            'token': token_str,
            'mem_score': float(mem_score)
        })
        
        # 5. Determine next token for the next step
        next_token_logits = outputs.logits[0, -1, :]
        next_token_id = torch.argmax(next_token_logits).item()
        
    return results

def main():
    parser = argparse.ArgumentParser(description="LLM Probe Live Inference CLI")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt to generate from")
    parser.add_argument("--max_tokens", type=int, default=100, help="Maximum number of tokens to generate")
    parser.add_argument("--probe_path", type=str, default="results/memorization/probe.joblib", help="Path to trained probe.joblib")
    args = parser.parse_args()
    
    print(f"Loading probe payload from: {args.probe_path}...")
    payload = load_probe_payload(args.probe_path)
    
    probe = payload['probe']
    scaler = payload['scaler']
    selected_layers = payload['selected_layers']
    representation_type = payload['representation_type']
    
    print(f"Loaded probe configuration: layers={selected_layers}, representation={representation_type}")
    print(f"Loading LLM model '{MODEL_NAME}' on '{DEVICE}'...")
    model = load_model(MODEL_NAME, DEVICE)
    tokenizer = load_tokenizer(MODEL_NAME)
    
    print(f"\nGenerating and scoring (max_tokens={args.max_tokens})...")
    results = generate_with_probe(
        prompt=args.prompt,
        model=model,
        tokenizer=tokenizer,
        probe=probe,
        scaler=scaler,
        selected_layers=selected_layers,
        representation_type=representation_type,
        max_tokens=args.max_tokens
    )
    
    # Import and render results via terminal
    from demo.render import render_terminal
    render_terminal(results)

if __name__ == "__main__":
    main()
