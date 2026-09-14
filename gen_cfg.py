import re
import json
import os

BASE = '/home/zll2025/low-rank-cl-main/configs'

# value tokens as used in imagenet-a file names, in canonical order
TOKENS = ['0.0000001','0.000001','0.00001','0.0001','0.001','0.01','0.1','1','10','100']
NUM = {t: float(t) for t in TOKENS}

# dir, stem(dataset name in filename), seed-token in filename, template filename, skip values already present
SPECS = [
    ('sdlora_ov4',  'cifar100',  '1993', 'cifar100_seed=1993_lo=100_ls_100_mu=100.json',                  {'100'}),
    ('sdlora_ov4',  'imagenet-r','1993', 'imagenet-r_seed=1993_lo=100_ls_100_mu=100.json',                {'100'}),
    ('sdlora_ov4',  'domain',    '1993', 'domain_seed=1993__lo=100_ls_100_mu=100.json',                   set()),   # existing file name has stray "__"
    ('splitlorav64','cifar100',  '1993', 'cifar100_seed=1993_lo=0.1_ls_0.1_mu=0.1.json',                  {'0.1'}),
    ('splitlorav64','imagenet-r','1993', 'imagenet-r_seed=1993_lo=0.1_ls_0.1_mu=0.1.json',                {'0.1'}),
    ('splitlorav64','domain',    '1993', 'domain_seed=1993_lo=0.1_ls_0.1_mu=0.1.json',                    {'0.1'}),
    ('ewclora_ov4', 'cifar100',  '1993', 'cifar100_seed=1993_lo=0.1_ls_0.1_mu=0.1.json',                  {'0.01','0.1'}),
    ('ewclora_ov4', 'imagenet-r','1993', 'imagenet-r_seed=1993_ro=0.1_lo=0.1_ls_0.1_mu=0.1.json',        set()),   # existing has stale "ro=" segment
    ('ewclora_ov4', 'domain',    '1993', 'domain_seed=1993_ro=0.09_ro=0.11_lo=0.1_ls_0.1_mu=0.1.json',   set()),   # old-schema file, append lambda keys
]

def sub_val(text, key, tok):
    pat = re.compile(r'("' + re.escape(key) + r'"\s*:\s*)[^,\n]+')
    return pat.sub(lambda m: m.group(1) + tok, text)

def replace_mode(text, tok):
    for key in ('lambda_o','lambda_s','mu'):
        text = sub_val(text, key, tok)
    return text

def append_mode(text, tok):
    # ensure last line before '}' has comma, then append the 3 lambda keys
    text = text.rstrip()
    if not text.endswith('}'):
        raise SystemExit('append_mode: file does not end with }')
    head = text[:-1].rstrip()
    # add comma to final key line if missing
    head = re.sub(r'(\n[ \t]*"[^"]+"[ \t]*:[ \t]*[^,\n]+)\s*$', r'\1,', head)
    return head + ('\n    "lambda_o": %s,\n    "lambda_s": %s,\n    "mu": %s\n}' % (tok, tok, tok))

def main():
    created, skipped, errors = [], [], []
    for d, stem, seed, tpl, skip in SPECS:
        tpl_path = os.path.join(BASE, d, tpl)
        with open(tpl_path) as f:
            raw = f.read()
        # determine whether template already carries lambda_o/lambda_s/mu
        has_all = all(re.search(r'"%s"\s*:' % k, raw) for k in ('lambda_o','lambda_s','mu'))
        for tok in TOKENS:
            if tok in skip:
                continue
            fname = '%s_seed=%s_lo=%s_ls_%s_mu=%s.json' % (stem, seed, tok, tok, tok)
            out = os.path.join(BASE, d, fname)
            if os.path.exists(out):
                skipped.append(out)
                continue
            if has_all:
                content = replace_mode(raw, tok)
            else:
                content = append_mode(raw, tok)
            # validate JSON + numeric equality
            data = json.loads(content)
            if has_all:
                assert float(data['lambda_o']) == NUM[tok] and float(data['lambda_s']) == NUM[tok] and float(data['mu']) == NUM[tok], (out, data.get('lambda_o'))
            else:
                assert float(data['lambda_o']) == NUM[tok] and float(data['lambda_s']) == NUM[tok] and float(data['mu']) == NUM[tok], (out,)
            with open(out, 'w') as f:
                f.write(content)
            created.append(os.path.relpath(out, BASE))

    print('CREATED %d files:' % len(created))
    for c in created:
        print('  +', c)
    print('SKIPPED %d (already exist):' % len(skipped))
    for s in skipped:
        print('  =', s)

if __name__ == '__main__':
    main()