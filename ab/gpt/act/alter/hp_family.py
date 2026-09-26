"""
Generation-only entry point: one architecture shown under k hyperparameter
settings, with or without its train_stat diagnostics.

No fine-tuning and no LoRA — the model is loaded once and prompted. Output
layout matches AlterNN so the existing evaluator consumes it unchanged:

    <epoch_dir>/<round>/synth_nn/B<i>/{new_nn.py, full_output.txt,
                                       original_<nn>.py, dataframe.df}

Two arms, differing only in the config's hp_family.show_diagnostics:

    python -m ab.gpt.act.alter.hp_family --arm control      --rounds 3
    python -m ab.gpt.act.alter.hp_family --arm experimental --rounds 3

Secondary pair (epoch 1, transform held constant, 183 families):

    python -m ab.gpt.act.alter.hp_family --arm control-e1      --rounds 3
    python -m ab.gpt.act.alter.hp_family --arm experimental-e1 --rounds 3

Then train what was generated (3 epochs) and tag it so the arms stay separable:

    python -m ab.gpt.act.eval.Eval --nn_train_epochs 3 --nn_name_prefix hpf-ctl

--dry-run assembles and prints the prompts without loading a model (CPU only).
"""
import argparse
import json
import os
import shutil

import torch
from tqdm import tqdm

from ab.nn.util.Util import create_file
from ab.gpt.util.Const import conf_test_dir, epoch_dir, new_nn_file, new_out_file, synth_dir
from ab.gpt.util.Util import extract_code
from ab.gpt.util.hp_family import (
    DEFAULT_DIAGNOSTIC_FIELDS, fetch_families, render_family_block, anchor_row)

# Primary pair: epoch 10, grouped by architecture alone (members carry their own
# transform), member accuracy shown in both arms — the diagnostics decouple from
# accuracy there. The '-e1' pair is the epoch-1 high-n fallback.
ARM_CONF = {
    'control': ('NN_gen_hp_family_control.json', 'hp_control'),
    'experimental': ('NN_gen_hp_family_experimental.json', 'hp_experimental'),
    'control-v2': ('NN_gen_hp_family_control_v2.json', 'hp_control_v2'),
    'experimental-v2': ('NN_gen_hp_family_experimental_v2.json', 'hp_experimental_v2'),
    'control-e1': ('NN_gen_hp_family_control_epoch1.json', 'hp_control_e1'),
    'experimental-e1': ('NN_gen_hp_family_experimental_epoch1.json', 'hp_experimental_e1'),
}


def build_prompts(conf_file, conf_key=None):
    """Assemble (system_text, prompt_text, origdf) per family. No model needed."""
    with open(conf_test_dir / conf_file) as f:
        prompt_dict = json.load(f)
    keys = [conf_key] if conf_key else list(prompt_dict)

    prompts = []
    for key in keys:
        cfg = prompt_dict[key]
        hpf = cfg.get('hp_family') or {}
        fields = tuple(hpf.get('diagnostic_fields') or DEFAULT_DIAGNOSTIC_FIELDS)
        system_text = '\n'.join(cfg.get('system', []))
        template = ''.join(p + '\n' for p in cfg['prompt'])

        families = fetch_families(
            task=cfg.get('task', 'img-classification'),
            dataset=cfg.get('dataset', 'cifar-10'),
            metric=cfg.get('metric', 'acc'),
            epoch=hpf.get('epoch', 1),
            min_settings=hpf.get('min_settings', 3),
            max_members=hpf.get('max_members', 4),
            min_lr_ratio=hpf.get('min_lr_ratio', 2.0),
            diagnostic_fields=fields,
            nn_prefixes=tuple(hpf.get('nn_prefixes') or ()) or None,
            max_families=hpf.get('max_families'),
            group_by_transform=hpf.get('group_by_transform', True),
        )
        print(f'[{key}] epoch={hpf.get("epoch", 1)} '
              f'group_by_transform={hpf.get("group_by_transform", True)} '
              f'families={len(families)} '
              f'(k: {min((f.k for f in families), default=0)}-{max((f.k for f in families), default=0)}) '
              f'show_diagnostics={hpf.get("show_diagnostics", True)}')

        for fam in families:
            para = {it['para']: getattr(fam, it['value'], None) for it in cfg['input_list']}
            para['hp_family_prompt'] = render_family_block(
                fam, diagnostic_fields=fields,
                show_diagnostics=hpf.get('show_diagnostics', True),
                show_member_accuracy=hpf.get('show_member_accuracy', True))
            prompts.append((system_text, template.format(**para), anchor_row(fam)))
    return prompts


def generate(prompts, llm_name, rounds, temperature, top_k, max_new_tokens, out_root=None):
    from ab.gpt.util.LLM import LLM

    model_loader = LLM(llm_name, load_in_4bit=False)
    model, tokenizer = model_loader.get_model(), model_loader.get_tokenizer()
    root = out_root or epoch_dir()
    shutil.rmtree(root, ignore_errors=True)

    for rnd in range(rounds):
        out_path = epoch_dir(rnd)
        b_index = 0
        for system_text, prompt_text, origdf in tqdm(prompts, desc=f'Round {rnd}'):
            msgs = ([{'role': 'system', 'content': system_text}] if system_text else [])
            msgs.append({'role': 'user', 'content': prompt_text})
            inputs = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, return_tensors='pt')
            inputs = (inputs.input_ids if hasattr(inputs, 'input_ids') else inputs).to(model.device)

            with torch.no_grad():
                model.eval()
                out_ids = model.generate(
                    inputs, max_new_tokens=max_new_tokens, do_sample=True,
                    temperature=temperature, top_k=top_k, top_p=0.95,
                    num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
            out = tokenizer.decode(out_ids[0][len(inputs[0]):], skip_special_tokens=True)

            nn_code = extract_code(out)
            if not nn_code:
                print('[INFO] Response invalid, skipping')
                continue
            model_dir = synth_dir(out_path) / f'B{b_index}'
            model_dir.mkdir(exist_ok=True, parents=True)
            (model_dir / new_nn_file).write_text(nn_code)
            create_file(model_dir, new_out_file, out)
            (model_dir / f"original_{origdf['nn']}.py").write_text(origdf['nn_code'])
            origdf.to_pickle(model_dir / 'dataframe.df')
            b_index += 1
        print(f'[round {rnd}] saved {b_index} models under {synth_dir(out_path)}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--arm', choices=sorted(ARM_CONF), default='experimental')
    ap.add_argument('--conf', help='override the config file name')
    ap.add_argument('--conf-key', help='override the config key')
    ap.add_argument('--rounds', type=int, default=1, help='generation rounds over all families')
    ap.add_argument('--llm', default='deepseek-ai/deepseek-coder-7b-instruct-v1.5')
    ap.add_argument('--temperature', type=float, default=0.6)
    ap.add_argument('--top_k', type=int, default=50)
    ap.add_argument('--max_new_tokens', type=int, default=4096)
    ap.add_argument('--dry-run', action='store_true',
                    help='assemble prompts and print the first one; no model is loaded')
    ap.add_argument('--show', type=int, default=1, help='prompts to print with --dry-run')
    args = ap.parse_args()

    conf_file, conf_key = ARM_CONF[args.arm]
    prompts = build_prompts(args.conf or conf_file, args.conf_key or conf_key)
    print(f'[{args.arm}] assembled {len(prompts)} prompts')

    if args.dry_run:
        for i, (system_text, text, origdf) in enumerate(prompts[:args.show]):
            print(f'\n{"=" * 78}\nPROMPT {i}  (family {origdf["nn"]} / {origdf["transform"]})'
                  f'  {len(text)} chars\n{"=" * 78}')
            if system_text:
                print(f'[system] {system_text}\n')
            print(text)
        return
    generate(prompts, args.llm, args.rounds, args.temperature, args.top_k, args.max_new_tokens)


if __name__ == '__main__':
    main()
