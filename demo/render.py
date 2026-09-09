"""
CLI Rendering Utilities for LLM Probe Live Inference.
Provides ANSI color-coded token display and dual-probe execution summaries.
"""

# ANSI Color Codes
CYAN = "\033[96m"
BOLD_CYAN = "\033[1;96m"
RED = "\033[91m"
BOLD_RED = "\033[1;91m"
YELLOW = "\033[93m"
BOLD_YELLOW = "\033[1;93m"
GREEN = "\033[92m"
BOLD_GREEN = "\033[1;92m"
GRAY = "\033[90m"
BOLD = "\033[1m"
RESET = "\033[0m"


def get_risk_level_str(max_score):
    """Returns ANSI formatted risk level string based on max score."""
    if max_score >= 0.6:
        return f"{BOLD_RED}HIGH{RESET}"
    elif max_score >= 0.3:
        return f"{BOLD_YELLOW}MEDIUM{RESET}"
    else:
        return f"{BOLD_GREEN}LOW{RESET}"


def format_colored_score(tag, score):
    """Formats a single score with ANSI color coding."""
    if score >= 0.6:
        return f"{BOLD_RED}{tag}:{score:.2f}{RESET}"
    elif score >= 0.3:
        return f"{BOLD_YELLOW}{tag}:{score:.2f}{RESET}"
    else:
        return f"{GREEN}{tag}:{score:.2f}{RESET}"


def render_terminal(results, task="both", token_counts=None):
    """
    Renders generated tokens with inline ANSI risk coloring and a detailed summary table.
    Supports single probe (memorization or hallucination) or dual probes simultaneously.

    Args:
        results: list of dicts, each with 'token', and 'mem_score' and/or 'halluc_score'
        task: str, 'both', 'memorization', or 'hallucination'
        token_counts: dict with 'prompt_tokens', 'generated_tokens', 'total_tokens', 'max_tokens'
    """
    if not results:
        print(f"{YELLOW}No tokens generated.{RESET}")
        return

    # Check which scores are present in results
    has_mem = any("mem_score" in item or "score" in item for item in results) and task != "hallucination"
    has_halluc = any("halluc_score" in item for item in results) and task != "memorization"
    is_dual = has_mem and has_halluc

    if is_dual:
        header_tag = "Memorization [M] | Hallucination [H]"
    elif has_halluc:
        header_tag = "Hallucination [HALLUC]"
    else:
        header_tag = "Memorization [MEM]"

    print(f"\n{BOLD_CYAN}=== Generated Text with Inline Per-Token Risk ({header_tag}) ==={RESET}\n")

    mem_scores = []
    halluc_scores = []

    for item in results:
        token = item.get("token", "")
        mem_s = item.get("mem_score", item.get("score"))
        halluc_s = item.get("halluc_score")

        if mem_s is not None and has_mem:
            mem_scores.append(mem_s)
        if halluc_s is not None and has_halluc:
            halluc_scores.append(halluc_s)

        if is_dual:
            tag_str = (
                f"{GRAY}[{RESET}"
                f"{format_colored_score('M', mem_s)}"
                f"{GRAY}|{RESET}"
                f"{format_colored_score('H', halluc_s)}"
                f"{GRAY}]{RESET}"
            )
        elif has_halluc and halluc_s is not None:
            tag_str = f"{GRAY}[{RESET}{format_colored_score('HALLUC', halluc_s)}{GRAY}]{RESET}"
        elif mem_s is not None:
            tag_str = f"{GRAY}[{RESET}{format_colored_score('MEM', mem_s)}{GRAY}]{RESET}"
        else:
            tag_str = ""

        print(f"{token}{tag_str}", end="")
    print("\n")

    # Summary Card
    width = 58
    print(f"{CYAN}{'=' * width}{RESET}")
    print(f"{BOLD_CYAN}{'INFERENCE & PROBE SUMMARY':^{width}}{RESET}")
    print(f"{CYAN}{'=' * width}{RESET}")

    # Token Count Section
    gen_cnt = len(results)
    if token_counts:
        gen_cnt = token_counts.get("generated_tokens", len(results))
        max_limit = token_counts.get("max_tokens")
        prompt_cnt = token_counts.get("prompt_tokens")

        limit_info = f" (Token Limit: {max_limit})" if max_limit else ""
        if max_limit and gen_cnt >= max_limit:
            limit_info = f" (Token Limit: {max_limit} [Limit Reached])"
        elif max_limit and gen_cnt < max_limit:
            limit_info = f" (Token Limit: {max_limit} [EOS Reached])"

        print(f"  {BOLD}Generated Answer{RESET}   : {BOLD_CYAN}{gen_cnt} tokens{RESET}{limit_info}")
        if prompt_cnt is not None:
            print(f"  {BOLD}Prompt Length{RESET}      : {prompt_cnt} tokens")
    else:
        print(f"  {BOLD}Generated Answer{RESET}   : {BOLD_CYAN}{gen_cnt} tokens{RESET}")

    # Memorization Probe Stats
    if has_mem and mem_scores:
        mem_max = max(mem_scores)
        mem_avg = sum(mem_scores) / len(mem_scores)
        print(f"{GRAY}{'-' * width}{RESET}")
        print(f"  {BOLD}PROBE: MEMORIZATION{RESET}")
        print(f"    - Risk Level     : {get_risk_level_str(mem_max)}")
        print(f"    - Max Token Score: {mem_max:.4f}")
        print(f"    - Avg Token Score: {mem_avg:.4f}")

    # Hallucination Probe Stats
    if has_halluc and halluc_scores:
        halluc_max = max(halluc_scores)
        halluc_avg = sum(halluc_scores) / len(halluc_scores)
        print(f"{GRAY}{'-' * width}{RESET}")
        print(f"  {BOLD}PROBE: HALLUCINATION{RESET}")
        print(f"    - Risk Level     : {get_risk_level_str(halluc_max)}")
        print(f"    - Max Token Score: {halluc_max:.4f}")
        print(f"    - Avg Token Score: {halluc_avg:.4f}")

    print(f"{CYAN}{'=' * width}{RESET}\n")
