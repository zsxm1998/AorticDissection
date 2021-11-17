import os

ckp_dir = 'details/checkpoints'
for name in os.listdir(ckp_dir):
    file = os.path.join(ckp_dir, name)
    if os.path.isfile(file) and file.endswith('_INTERRUPTED.pth'):
        os.remove(file)