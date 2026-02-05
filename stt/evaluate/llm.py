import json
import os
import re
import torch
from .honest_llama.utils import alt_tqa_evaluate

def evaluate_clutrr(eval_dataset, model_name, model, tokenizer, fname, batch_size=16, max_new_tokens=16,
                    apply_chat_template=False):  

    results_dir = os.path.join(fname, 'outputs.json')
    result_json = []
    tokenizer.padding_side = 'left'
    inputs = eval_dataset['prompt']
    iterator = range(0, len(inputs), batch_size)
    generated = []

    def normalize_prediction(ans: str) -> str:

        ans = ans.lower()
        ans = re.sub(r"</s>|<pad>|<\|endoftext\|>|[<>]", " ", ans)
        ans = re.sub(r"\s+", " ", ans).strip()

        ans = re.sub(
            r"(father-in-law|mother-in-law|grandmother|grandfather|granddaughter|grandson|"
            r"father|mother|daughter|son|brother|sister|uncle|aunt|nephew|niece)"
            r"\1+",
            r"\1",
            ans,
        )

        rel_pattern = re.compile(
            r"\b("
            r"father-in-law|mother-in-law|granddaughter|grandson|"
            r"grandmother|grandfather|daughter|mother|brother|sister|father|"
            r"nephew|niece|uncle|aunt|son"
            r")\b"
        )
        match = rel_pattern.search(ans)
        if match:
            return match.group(1)

        return ans.split()[0] if ans else ""

    with torch.no_grad():
        for i in iterator:
            inputs_b = inputs[i:i + batch_size]
            inputs_b = tokenizer(inputs_b, return_tensors='pt', padding=True, truncation=True, max_length=512)
            inputs_b = {k: v.to(model.device) for (k, v) in inputs_b.items()}
            outputs = model.generate(**inputs_b, max_new_tokens=max_new_tokens, do_sample=False, use_cache=False)
            decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            generated.extend(decoded_outputs)

    corr = 0
    for i in range(len(generated)):
        seq = generated[i]

        sep = '[/INST]\n' if apply_chat_template else "'s\n"
        try:
            ans_raw = seq.split(sep)[1].split('.')[0]
        except IndexError:
            ans_raw = seq

        pred = normalize_prediction(ans_raw)
        label = eval_dataset['target_text'][i].lower().strip()  
        entry_corr = int(pred == label)
        corr += entry_corr

        result_json.append({
            'prompt': inputs[i],
            'response': seq,
            'pred': pred,
            'label': label,
            'correct': entry_corr
        })

    accuracy = corr / len(generated)
    print(f"Accuracy: {accuracy:.4f}")
    with open(results_dir, 'w') as f:
        json.dump(result_json, f, indent=2)

    return accuracy, generated


def evaluate_tqa(eval_dataset, fname, model, tokenizer, metrics, model_name=None, verbose=False):
    ### Create directories to save truthfulqa outputs
    if not os.path.exists('./tqa_results/answer_dump'):
        os.makedirs('./tqa_results/answer_dump')
    if not os.path.exists('./tqa_results/summary_dump'):
        os.makedirs('./tqa_results/summary_dump')
    curr_fold_results = alt_tqa_evaluate(
        {model_name: model},
        metric_names=metrics,
        input_path=eval_dataset['data_dir'],
        output_path=f'./tqa_results/answer_dump/{model_name}_truthfulqa.csv',
        summary_path=f'./tqa_results/summary_dump/{model_name}_truthfulqa.csv',
        device="cuda",
        tokenizer=tokenizer,
        ### Print generated outputs
        verbose=verbose,
        ### Use the standard QA prompt for evaluation
        preset='qa'
    )
    print(curr_fold_results)

