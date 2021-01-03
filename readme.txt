train: python main.py --config res34_112x112_adam_lr1e-3.yaml
resume train:  python main.py --config res34_112x112_adam_lr1e-3.yaml --load ./checkpoint/ckpt_last.pth.tar --resume
test: python main.py --load ./checkpoint/best.pth --test