import argparse
import json
import re
import sys
import ollama

MODEL_NAME = "qwen3:8b"


def clean_model_text(text: str) -> str:
    """حذف تگ‌های تفکر درونی مدل در صورت وجود"""
    if not text:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


def call_llm(system_prompt: str, user_prompt: str) -> str:
    """فراخوانی محلی مدل از طریق Ollama بدون گیر افتادن در حلقه تفکر طولانی"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    options = {
        "temperature": 0.0,
        "num_predict": 1500
    }

    try:
        try:
            response = ollama.chat(
                model=MODEL_NAME,
                messages=messages,
                think=False,
                options=options
            )
        except TypeError:
            # سازگاری با نسخه‌های قدیمی‌تر کتابخانه در صورت نیاز
            messages[-1]["content"] += "\n/no_think"
            response = ollama.chat(
                model=MODEL_NAME,
                messages=messages,
                options=options
            )

        msg = response.get("message", {})
        raw_content = msg.get("content", "") or ""
        thinking_content = msg.get("thinking", "") or ""

        processed = clean_model_text(raw_content)
        if processed:
            return processed
        return clean_model_text(thinking_content)

    except Exception as e:
        print(f"[Error] خطا در ارتباط با Ollama: {e}", file=sys.stderr)
        return ""


def extract_tag_or_last_line(text: str) -> str:
    """استخراج مقدار داخل تگ FINAL_ANSWER یا آخرین خط معتبر"""
    if not text or not text.strip():
        return ""

    matches = re.findall(r"<FINAL_ANSWER>(.*?)</FINAL_ANSWER>", text, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        for m in reversed(matches):
            candidate = m.strip()
            if candidate:
                return candidate

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def clean_final_answer(ans: str) -> str:
    """پاکسازی کامل پاسخ نهایی از هرگونه Markdown یا عبارت اضافی"""
    ans = ans.replace("`", "").replace("**", "").replace('"', "").replace("'", "").strip()
    ans = re.sub(r"^<FINAL_ANSWER>|<\/FINAL_ANSWER>$", "", ans, flags=re.IGNORECASE).strip()
    ans = re.sub(r"^(پاسخ نهایی|پاسخ|جواب نهایی|جواب|Final Answer|Answer)\s*[:：\-=]\s*", "", ans, flags=re.IGNORECASE).strip()
    lines = [line.strip() for line in ans.splitlines() if line.strip()]
    return lines[-1] if lines else ans


def extract_final_answer(verifier_output: str, draft_solution: str) -> str:
    """استخراج پاسخ از خروجی بازبین و استفاده از خروجی حل‌کننده به عنوان پشتیبان"""
    ans = extract_tag_or_last_line(verifier_output)
    if not ans:
        ans = extract_tag_or_last_line(draft_solution)
    return clean_final_answer(ans)


def solve_problem_agent(problem_text: str) -> str:
    """معماری دو مرحله‌ای Agent: ۱. برنامه‌ریز و حل‌کننده ۲. بازبین و قالب‌بند"""

    # مرحله ۱: Planner & Solver
    planner_system = (
        "You are an expert logical and mathematical Planner and Solver agent. "
        "Break down the problem into constraints, solve it concisely step-by-step, "
        "and if multiple valid answers exist, choose the most direct one immediately. "
        "Always end your response with <FINAL_ANSWER>your_answer</FINAL_ANSWER>."
    )
    planner_user = (
        f"Problem:\n{problem_text}\n\n"
        "Instructions:\n"
        "1. List the key constraints and variables briefly.\n"
        "2. Solve step-by-step clearly and concisely.\n"
        "3. Write the exact final answer matching the requested format inside <FINAL_ANSWER>...</FINAL_ANSWER> tags."
    )
    draft_solution = call_llm(planner_system, planner_user)

    # مرحله ۲: Verifier & Formatter
    verifier_system = (
        "You are a strict Verifier and Output Formatter agent. "
        "Check the proposed solution against the problem constraints and required output format. "
        "If there is a clear mathematical or constraint error, fix it. "
        "Finally, output the exact final answer inside <FINAL_ANSWER>...</FINAL_ANSWER> tags."
    )
    verifier_user = (
        f"Original Problem:\n{problem_text}\n\n"
        f"Proposed Solution:\n{draft_solution}\n\n"
        "Instructions:\n"
        "1. Verify correctness and format in at most 2-3 short lines.\n"
        "2. Place ONLY the exact required final answer inside <FINAL_ANSWER>...</FINAL_ANSWER> tags with no extra text."
    )
    verified_output = call_llm(verifier_system, verifier_user)

    return extract_final_answer(verified_output, draft_solution)


def main():
    parser = argparse.ArgumentParser(description="Local Reasoning LLM Agent with Ollama")
    parser.add_argument("--input", required=True, help="Path to input.json")
    parser.add_argument("--output", required=True, help="Path to output.json")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    if isinstance(input_data, dict) and "problem" in input_data:
        problem_id = input_data.get("id", "test_001")
        print(f"در حال حل مسئله {problem_id} ...")
        ans = solve_problem_agent(input_data["problem"])
        print(f"پاسخ استخراج شده برای {problem_id}: {ans}")
        output_data = {"id": problem_id, "answer": ans}

    elif isinstance(input_data, list):
        output_data = []
        for item in input_data:
            problem_id = item.get("id", "unknown")
            print(f"در حال حل مسئله {problem_id} ...")
            ans = solve_problem_agent(item.get("problem", ""))
            print(f"پاسخ استخراج شده برای {problem_id}: {ans}")
            output_data.append({"id": problem_id, "answer": ans})
    else:
        raise ValueError("فرمت فایل ورودی JSON نامعتبر است.")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\nتمامی پاسخ‌ها با موفقیت در فایل {args.output} ذخیره شدند.")


if __name__ == "__main__":
    main()