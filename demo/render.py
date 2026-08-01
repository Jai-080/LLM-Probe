import os

def render_terminal(results):
    """
    Renders results in the terminal with ANSI colors.
    """
    if not results:
        print("No tokens generated.")
        return
        
    RED = "\033[91m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    RESET = "\033[0m"
    
    print("\n=== Generated Text with Inline Memorization Risk ===")
    for item in results:
        token = item['token']
        score = item['mem_score']
        
        # Color mapping based on score
        if score >= 0.6:
            color = RED
        elif score >= 0.3:
            color = YELLOW
        else:
            color = GREEN
            
        # Display inline
        print(f"{token}{color}[MEM:{score:.2f}]{RESET}", end="")
    print()
    
    # Summary
    scores = [item['mem_score'] for item in results]
    max_score = max(scores)
    avg_score = sum(scores) / len(scores)
    
    if max_score >= 0.6:
        risk_level = f"{RED}HIGH{RESET}"
    elif max_score >= 0.3:
        risk_level = f"{YELLOW}MEDIUM{RESET}"
    else:
        risk_level = f"{GREEN}LOW{RESET}"
        
    print("\n=== Response Summary ===")
    print(f"- Memorization risk: {risk_level} (max token score: {max_score:.4f}, avg score: {avg_score:.4f})")
    print("="*50 + "\n")

def get_color_for_score(score):
    """
    Calculates HSL color values based on the score (gradient from green to yellow to red).
    Green is Hue=120, Red is Hue=0.
    """
    # Clip score to [0.0, 1.0]
    score = max(0.0, min(1.0, score))
    # Map score to Hue (120 -> 0)
    hue = int(120 * (1 - score))
    # Low score = low opacity (soft highlight), High score = high opacity (vibrant highlight)
    opacity = 0.2 + (0.6 * score)
    return f"hsla({hue}, 85%, 50%, {opacity:.2f})", f"hsla({hue}, 85%, 35%, 0.8)"

def render_html(results, output_path):
    """
    Generates a standalone, beautiful HTML report showing color-coded tokens.
    """
    if not results:
        return
        
    scores = [item['mem_score'] for item in results]
    max_score = max(scores)
    
    if max_score >= 0.6:
        risk_text = "HIGH"
        risk_color = "#ef4444"
    elif max_score >= 0.3:
        risk_text = "MEDIUM"
        risk_color = "#eab308"
    else:
        risk_text = "LOW"
        risk_color = "#22c55e"
        
    # Generate inline spans for tokens
    token_spans = []
    for item in results:
        token = item['token']
        score = item['mem_score']
        bg_color, border_color = get_color_for_score(score)
        
        # Escape HTML special characters
        escaped_token = token.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        
        span = (
            f'<span style="background-color: {bg_color}; border-bottom: 2px solid {border_color}; '
            f'padding: 1px 3px; margin: 0 1px; border-radius: 2px; cursor: help; display: inline-block;" '
            f'title="Memorization Score: {score:.4f}">{escaped_token}</span>'
        )
        token_spans.append(span)
        
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>LLM Probe - Memorization Inference Report</title>
    <style>
        body {{
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            background-color: #0f172a;
            color: #f8fafc;
            padding: 40px;
            margin: 0;
            display: flex;
            justify-content: center;
        }}
        .container {{
            max-width: 800px;
            width: 100%;
            background-color: #1e293b;
            border-radius: 12px;
            padding: 30px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1);
        }}
        h1 {{
            font-size: 24px;
            margin-top: 0;
            border-bottom: 1px solid #334155;
            padding-bottom: 15px;
            color: #38bdf8;
        }}
        .summary {{
            background-color: #0f172a;
            border-radius: 8px;
            padding: 15px 20px;
            margin-bottom: 25px;
            border-left: 4px solid {risk_color};
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .summary-label {{
            font-size: 14px;
            color: #94a3b8;
        }}
        .summary-value {{
            font-size: 18px;
            font-weight: 700;
            color: {risk_color};
        }}
        .text-display {{
            background-color: #0f172a;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 20px;
            white-space: pre-wrap;
            line-height: 2.0;
            font-size: 16px;
        }}
        .footer {{
            margin-top: 30px;
            font-size: 12px;
            color: #64748b;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>LLM Probe - Per-Token Memorization Analysis</h1>
        <div class="summary">
            <div>
                <div class="summary-label">OVERALL MEMORIZATION RISK</div>
                <div class="summary-value" style="font-size: 24px;">{risk_text}</div>
            </div>
            <div>
                <div class="summary-label" style="text-align: right;">MAX TOKEN RISK SCORE</div>
                <div class="summary-value" style="text-align: right;">{max_score:.4f}</div>
            </div>
        </div>
        
        <div class="text-display">{''.join(token_spans)}</div>
        
        <div class="footer">
            Generated by LLM Probe • Growing-window mean-pool activation classification
        </div>
    </div>
</body>
</html>
"""
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Saved HTML analysis report to: {output_path}")
