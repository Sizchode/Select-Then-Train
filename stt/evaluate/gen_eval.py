import os
import json
import torch
from tqdm import tqdm
import re

def evaluate_multiple_choice(eval_dataset, model_name, model, tokenizer, fname, task_name, batch_size=16,
                             max_new_tokens=8, apply_chat_template=False):
    os.makedirs(fname, exist_ok=True)
    results_dir = os.path.join(fname, f'{task_name}_outputs.json')
    results_json = []
    tokenizer.padding_side = 'left'
    inputs = eval_dataset['prompt']

    iterator = range(0, len(inputs), batch_size)
    generated = []

    device = model.device if hasattr(model, 'device') else 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    with torch.no_grad():
        for i in tqdm(iterator, desc=f"Evaluating {task_name}"):
            inputs_b = inputs[i:i + batch_size]
            inputs_b = tokenizer(inputs_b, return_tensors='pt', padding=True, truncation=True, max_length=1024)
            inputs_b = {k: v.to(device) for (k, v) in inputs_b.items()}
            outputs = model.generate(**inputs_b, max_new_tokens=max_new_tokens, do_sample=False)
            decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            generated.extend(decoded_outputs)

    corr = 0
    for i in range(len(generated)):
        # seq = generated[i]
        seq = generated[i]
        if seq is None:
            seq = ""
        # stop at first end token
        if '</s>' in seq:
            seq = seq.split('</s>', 1)[0]

        # sep = '[/INST]\n' if apply_chat_template else 'Answer:'
        # if sep in seq:
        #     ans = seq.split(sep)[-1].strip().split('\n')[0].strip()
        # else:
        #     ans = seq[-5:].strip()
        sep = '[/INST]\n' if apply_chat_template else 'Answer:'
        if sep in seq:
            ans = seq.split(sep, 1)[-1].strip().split('\n')[0].strip()
        else:
            ans = seq.strip()

        ans = ans.replace('</s>', '').strip()
        if len(ans) > 1 and not ans.isalpha():
            for char in ans:
                if char.isalpha() or char.isdigit():
                    ans = char
                    break

        ans = ans.upper()
        gold = [label.upper() for label in eval_dataset['true_labels'][i]]

        entry_corr = int(ans in gold)
        corr += entry_corr
        results_json.append({
            'prompt': inputs[i],
            'response': seq,
            'pred': ans,
            'label': gold,
            'correct': entry_corr
        })

    accuracy = corr / len(generated)
    print(f"{task_name} Accuracy: {accuracy:.4f}")

    with open(results_dir, 'w') as f:
        json.dump(results_json, f, indent=2)

    return accuracy, results_json

def evaluate_multiple_choice_obqa(eval_dataset, model_name, model, tokenizer, fname, task_name, batch_size=16,
                             max_new_tokens=8, apply_chat_template=False):
    os.makedirs(fname, exist_ok=True)
    results_dir = os.path.join(fname, f'{task_name}_outputs.json')
    results_json = []
    tokenizer.padding_side = 'left'
    inputs = eval_dataset['prompt']

    iterator = range(0, len(inputs), batch_size)
    generated = []

    device = model.device if hasattr(model, 'device') else 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    with torch.no_grad():
        for i in tqdm(iterator, desc=f"Evaluating {task_name}"):
            inputs_b = inputs[i:i + batch_size]
            inputs_b = tokenizer(inputs_b, return_tensors='pt', padding=True, truncation=True, max_length=1024)
            inputs_b = {k: v.to(device) for (k, v) in inputs_b.items()}
            outputs = model.generate(**inputs_b, max_new_tokens=max_new_tokens, do_sample=False)
            decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            generated.extend(decoded_outputs)

    corr = 0
    # for i in range(len(generated)):
    #     seq = generated[i]
    #     sep = '[/INST]\n' if apply_chat_template else 'Answer:'
    #     match = re.search(r"\b([A-D])\b", seq)
    #     if match:
    #         ans = match.group(1)
    #     else:
            
    #         stripped = seq.strip()
    #         ans = stripped[0] if stripped else ""
    for i in range(len(generated)):
        seq = generated[i]
        sep = '[/INST]\n' if apply_chat_template else 'Answer:'
        # – old: search entire seq (matches prompt letters)
        # – match = re.search(r"\b([A-D])\b", seq)
        # – if match:
        # –     ans = match.group(1)
        # – else:
        # –     stripped = seq.strip()
        # –     ans = stripped[0] if stripped else ""
        # + isolate only the substring after the final separator
        if sep in seq:
            answer_section = seq.rsplit(sep, 1)[-1]
        else:
            answer_section = seq
        match = re.search(r"\b([A-D])\b", answer_section)
        if match:
            ans = match.group(1)
        else:
            stripped = answer_section.strip()
            ans = stripped[0] if stripped else ""


        gold = [label.upper() for label in eval_dataset['true_labels'][i]]

        entry_corr = int(ans in gold)
        corr += entry_corr
        results_json.append({
            'prompt': inputs[i],
            'response': seq,
            'pred': ans,
            'label': gold,
            'correct': entry_corr
        })

    accuracy = corr / len(generated)
    print(f"{task_name} Accuracy: {accuracy:.4f}")

    with open(results_dir, 'w') as f:
        json.dump(results_json, f, indent=2)

    return accuracy, results_json


# ----------------- HELPERS -----------------
_YES_PAT = r'\b(yes|true|yep|yeah|correct|right)\b'
_NO_PAT  = r'\b(no|false|nope|incorrect|wrong)\b'

_CLEAN_PAT = re.compile(
    r'(?:</s>|<pad>|<\s*/?\s*extra_id_\d+\s*>|<\|endoftext\|>)',
    flags=re.I,
)

def _extract_yes_no(text: str) -> str:
    """
    Extract YES / NO, else 'UNK'
    """
    # 1) Pickt the last "A:" 之后（兼容大小写）
    idx = text.lower().rfind('\na:')
    answer = text[idx + 3:] if idx != -1 else text

    # 2) delete extra token，Keep first five words
    answer = _CLEAN_PAT.sub(' ', answer)
    answer = " ".join(answer.strip().split())
    answer = " ".join(answer.split()[:5])

    # 3) Reg Expression
    if re.search(_YES_PAT, answer, flags=re.I):
        return "YES"
    if re.search(_NO_PAT, answer,  flags=re.I):
        return "NO"
    return "UNK"


# ----------------- MAIN -----------------
def evaluate_boolq(
        eval_dataset,
        model_name,
        model,
        tokenizer,
        fname,
        batch_size: int = 16,
        max_new_tokens: int = 8,
        apply_chat_template: bool = False,
):
    """
    
    """
    os.makedirs(fname, exist_ok=True)
    results_dir = os.path.join(fname, 'boolq_outputs.json')
    results_json = []

    tokenizer.padding_side = 'left'
    inputs = eval_dataset['prompt']

    iterator = range(0, len(inputs), batch_size)
    generated = []

    device = model.device if hasattr(model, 'device') else (
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    model = model.to(device)

    # -------- gen --------
    model.eval()
    with torch.no_grad():
        for i in tqdm(iterator, desc="Evaluating BoolQ"):
            batch_prompts = inputs[i:i + batch_size]
            enc = tokenizer(
                batch_prompts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=1024
            ).to(device)

            outs = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
            decoded = tokenizer.batch_decode(outs, skip_special_tokens=True)
            generated.extend(decoded)

    # -------- analyze & stats --------
    corr = 0
    for i, seq in enumerate(generated):
        pred = _extract_yes_no(seq)
        gold = [lbl.upper() for lbl in eval_dataset['true_labels'][i]]

        is_corr = int(pred in gold)
        corr += is_corr

        results_json.append({
            'prompt': inputs[i],
            'response': seq,
            'pred': pred,
            'label': gold,
            'correct': is_corr
        })

    accuracy = corr / len(generated)
    print(f"BoolQ Accuracy: {accuracy:.4f}")

    with open(results_dir, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)

    return accuracy, results_json

def evaluate_rte(eval_dataset, model_name, model, tokenizer, fname, batch_size=16, max_new_tokens=8,
                 apply_chat_template=False):
    return evaluate_multiple_choice(
        eval_dataset, model_name, model, tokenizer, fname,
        task_name="rte", batch_size=batch_size,
        max_new_tokens=max_new_tokens, apply_chat_template=apply_chat_template
    )


def evaluate_hellaswag(eval_dataset, model_name, model, tokenizer, fname, batch_size=16, max_new_tokens=8,
                       apply_chat_template=False):
    return evaluate_multiple_choice(
        eval_dataset, model_name, model, tokenizer, fname,
        task_name="hellaswag", batch_size=batch_size,
        max_new_tokens=max_new_tokens, apply_chat_template=apply_chat_template
    )


def evaluate_winogrande(eval_dataset, model_name, model, tokenizer, fname, batch_size=16, max_new_tokens=8,
                        apply_chat_template=False):
    return evaluate_multiple_choice(
        eval_dataset, model_name, model, tokenizer, fname,
        task_name="winogrande", batch_size=batch_size,
        max_new_tokens=max_new_tokens, apply_chat_template=apply_chat_template
    )


def evaluate_arc(eval_dataset, model_name, model, tokenizer, fname, subset="easy", batch_size=16, max_new_tokens=8,
                 apply_chat_template=False):
    return evaluate_multiple_choice(
        eval_dataset, model_name, model, tokenizer, fname,
        task_name=f"arc-{subset}", batch_size=batch_size,
        max_new_tokens=max_new_tokens, apply_chat_template=apply_chat_template
    )



def evaluate_obqa(eval_dataset, model_name, model, tokenizer, fname, batch_size=16, max_new_tokens=8,
                  apply_chat_template=False):
    return evaluate_multiple_choice_obqa(
        eval_dataset, model_name, model, tokenizer, fname,
        task_name="obqa", batch_size=batch_size,
        max_new_tokens=max_new_tokens, apply_chat_template=apply_chat_template
    )


def evaluate_musique(eval_dataset, model_name, model, tokenizer, fname, batch_size=16, max_new_tokens=32,
                     apply_chat_template=False):
    results_dir = os.path.join(fname, 'outputs.json')
    results_json = []
    tokenizer.padding_side = 'left'
    inputs = eval_dataset['prompt']

    iterator = range(0, len(inputs), batch_size)
    generated = []

    with torch.no_grad():
        for i in iterator:
            inputs_b = inputs[i:i + batch_size]
            inputs_b = tokenizer(inputs_b, return_tensors='pt', padding=True, truncation=True, max_length=1024)
            inputs_b = {k: v.to(model.device) for (k, v) in inputs_b.items()}
            outputs = model.generate(**inputs_b, max_new_tokens=max_new_tokens, do_sample=False)
            decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            generated.extend(decoded_outputs)

    corr = 0
    for i in range(len(generated)):
        seq = generated[i]
        sep = '[/INST]\n' if apply_chat_template else 'A:'
        ans = seq.split(sep)[-1].split('\n')[0].strip().lower()
        ans = ans.replace('</s>', '') if '</s>' in ans else ans
        gold = [label.lower() for label in eval_dataset['true_labels'][i]]
        entry_corr = int(ans in gold)
        corr += entry_corr
        results_json.append({'prompt': inputs[i], 'response': seq, 'pred': ans, 'label': gold, 'correct': entry_corr})

    accuracy = corr / len(generated)
    print(f"MuSiQue Accuracy: {accuracy}")
    json.dump(results_json, open(results_dir, 'w'))
    return accuracy, generated


def evaluate_babi(eval_dataset, model_name, model, tokenizer, fname, batch_size=16, max_new_tokens=8,
                  apply_chat_template=False):
    results_dir = os.path.join(fname, 'outputs.json')
    results_json = []
    tokenizer.padding_side = 'left'
    inputs = eval_dataset['prompt']

    iterator = range(0, len(inputs), batch_size)
    generated = []

    with torch.no_grad():
        for i in iterator:
            inputs_b = inputs[i:i + batch_size]
            inputs_b = tokenizer(inputs_b, return_tensors='pt', padding=True, truncation=True, max_length=512)
            inputs_b = {k: v.to(model.device) for (k, v) in inputs_b.items()}
            outputs = model.generate(**inputs_b, max_new_tokens=max_new_tokens, do_sample=False)
            decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            generated.extend(decoded_outputs)

    corr = 0
    for i in range(len(generated)):
        seq = generated[i]
        sep = '[/INST]\n' if apply_chat_template else 'A:'
        ans = seq.split(sep)[-1].split('\n')[0].strip().lower()
        ans = ans.replace('</s>', '') if '</s>' in ans else ans
        gold = eval_dataset['answer'][i].lower()
        entry_corr = int(ans == gold)
        corr += entry_corr
        results_json.append({'prompt': inputs[i], 'response': seq, 'pred': ans, 'label': gold, 'correct': entry_corr})

    accuracy = corr / len(generated)
    print(f"bAbI Accuracy: {accuracy}")
    json.dump(results_json, open(results_dir, 'w'))
    return accuracy, generated


def evaluate_cogs(eval_dataset, model_name, model, tokenizer, fname, batch_size=16, max_new_tokens=16,
                  apply_chat_template=False):
    results_dir = os.path.join(fname, 'outputs.json')
    results_json = []
    tokenizer.padding_side = 'left'
    inputs = eval_dataset['prompt']

    iterator = range(0, len(inputs), batch_size)
    generated = []

    with torch.no_grad():
        for i in iterator:
            inputs_b = inputs[i:i + batch_size]
            inputs_b = tokenizer(inputs_b, return_tensors='pt', padding=True, truncation=True, max_length=512)
            inputs_b = {k: v.to(model.device) for (k, v) in inputs_b.items()}
            outputs = model.generate(**inputs_b, max_new_tokens=max_new_tokens, do_sample=False)
            decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            generated.extend(decoded_outputs)

    corr = 0
    for i in range(len(generated)):
        seq = generated[i]
        sep = '[/INST]\n' if apply_chat_template else 'A:'
        ans = seq.split(sep)[-1].split('\n')[0].strip().lower()
        ans = ans.replace('</s>', '') if '</s>' in ans else ans
        gold = eval_dataset['target_text'][i].lower()
        entry_corr = int(ans == gold)
        corr += entry_corr
        results_json.append({'prompt': inputs[i], 'response': seq, 'pred': ans, 'label': gold, 'correct': entry_corr})

    accuracy = corr / len(generated)
    print(f"COGS Accuracy: {accuracy}")
    json.dump(results_json, open(results_dir, 'w'))
    return accuracy, generated


def evaluate_gsm8k(eval_dataset, model, tokenizer, output_dir, batch_size=16, max_new_tokens=16):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'gsm8k_outputs.json')
    results = []

    tokenizer.padding_side = 'left'
    inputs = eval_dataset['prompt']
    targets = eval_dataset['true_labels']

    generated = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(inputs), batch_size):
            batch_inputs = inputs[i:i + batch_size]
            tokenized = tokenizer(batch_inputs, return_tensors='pt', padding=True, truncation=True)
            tokenized = {k: v.to(model.device) for k, v in tokenized.items()}
            outputs = model.generate(**tokenized, max_new_tokens=max_new_tokens, do_sample=False)
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            generated.extend(decoded)

    correct = 0
    for i, output in enumerate(generated):
        answer = output.split(eval_dataset['prompt'][i])[-1].strip().split('\n')[0]
        answer = answer.replace('</s>', '').strip()
        if answer in [label.strip() for label in targets[i]]:
            correct += 1
            is_correct = 1
        else:
            is_correct = 0
        results.append({
            'prompt': inputs[i],
            'response': output,
            'predicted': answer,
            'label': targets[i],
            'correct': is_correct
        })

    accuracy = correct / len(generated)
    print(f"GSM8K Accuracy: {accuracy:.4f}")

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    return accuracy, generated