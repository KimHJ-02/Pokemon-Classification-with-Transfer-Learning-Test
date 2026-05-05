import os
import subprocess
import shutil


EXPERIMENTS = [
    {"model": "resnet50", "pretrained": "True", "epochs": "5", "batch_size": "32"},
    {"model": "mobilenet_v2", "pretrained": "True", "epochs": "5", "batch_size": "32"},
    {"model": "efficientnet_b0", "pretrained": "True", "epochs": "5", "batch_size": "32"},
    {"model": "simplecnn", "pretrained": "False", "epochs": "5", "batch_size": "64"},
]


def ensure_dirs():
    os.makedirs('logs', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)


def run_and_tee(cmd, log_path):
    with open(log_path, 'w', encoding='utf-8') as lf:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        for line in process.stdout:
            print(line, end='')
            lf.write(line)
        process.wait()
        return process.returncode


def run_experiment(cfg):
    model = cfg['model']
    pretrained = cfg['pretrained']
    epochs = cfg['epochs']
    batch_size = cfg['batch_size']

    cmd = [
        'python', '-u', 'train.py',
        '--model', model,
        '--pretrained', pretrained,
        '--epochs', epochs,
        '--batch_size', batch_size,
        '--data_dir', 'data'
    ]

    log_path = os.path.join('logs', f'{model}.log')
    print('Running:', ' '.join(cmd))
    rc = run_and_tee(cmd, log_path)
    print(f'Finished {model}, exit {rc}, log: {log_path}')
    if rc != 0:
        print(f'Skipping evaluation for {model} due to train failure.')
        return

    # archive history
    if os.path.exists('history.csv'):
        dst = f'history_{model}.csv'
        shutil.move('history.csv', dst)
        print('Saved', dst)

    # run evaluation if checkpoint exists
    ckpt = os.path.join('checkpoints', f'{model}_best.pth')
    if os.path.exists(ckpt):
        eval_cmd = ['python', '-u', 'evaluate.py', '--checkpoint', ckpt, '--model', model, '--data_dir', 'data']
        eval_log = os.path.join('logs', f'eval_{model}.log')
        eval_rc = run_and_tee(eval_cmd, eval_log)
        print(f'Finished eval {model}, exit {eval_rc}, log: {eval_log}')
        # move classification report
        rep_src = 'classification_report.csv'
        if os.path.exists(rep_src):
            rep_dst = f'classification_report_{model}.csv'
            shutil.move(rep_src, rep_dst)
            print('Saved', rep_dst)


def main():
    ensure_dirs()
    for cfg in EXPERIMENTS:
        run_experiment(cfg)


if __name__ == '__main__':
    main()
