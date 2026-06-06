import torch
import os, sys
import os.path as osp
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.paths import datasets_path


class CIFAR10(object):

    def __init__(self, train=True, resolution=32):
        self.data, self.targets = torch.load(
            osp.join(datasets_path, f'cifar10{resolution}',
                     'train.pt' if train else 'eval.pt')
        )
        self.data = self.data.float() / 255.0

        if train:
            self.data, self.targets = self.data, self.targets
        else:
            self.data, self.targets = self.data, self.targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]


if __name__ == "__main__":
    import argparse
    from PIL import Image
    import numpy as np
    from torchvision import datasets as tv_datasets

    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--download", action="store_true", default=True)  # download to datasets_path/cifar10
    args = parser.parse_args()

    res = args.resolution
    out_dir = osp.join(datasets_path, f"cifar10{res}")
    os.makedirs(out_dir, exist_ok=True)

    train_set = tv_datasets.CIFAR10(
        root=osp.join(datasets_path, "cifar10"),
        train=True, download=args.download, transform=None
    )
    eval_set = tv_datasets.CIFAR10(
        root=osp.join(datasets_path, "cifar10"),
        train=False, download=args.download, transform=None
    )


    def to_chw_long(img_pil: Image.Image, size: int):
        if img_pil.size != (size, size):
            img_pil = img_pil.resize((size, size), resample=Image.BILINEAR)
        arr = np.array(img_pil)  # H,W,3
        chw = arr.transpose(2, 0, 1)  # 3,H,W
        return torch.LongTensor(chw)


    train_imgs = []
    train_labels = []
    for img, label in train_set:
        img_tensor = to_chw_long(img, res)
        train_imgs.append(img_tensor)
        train_labels.append(label)

    print(f"{len(train_imgs)} train images")

    train_imgs = torch.stack(train_imgs)
    train_labels = torch.LongTensor(train_labels)
    torch.save([train_imgs, train_labels], osp.join(out_dir, "train.pt"))

    eval_imgs = []
    eval_labels = []
    for img, label in eval_set:
        img_tensor = to_chw_long(img, res)
        eval_imgs.append(img_tensor)
        eval_labels.append(label)

    print(f"{len(eval_imgs)} eval images")

    eval_imgs = torch.stack(eval_imgs)  # [N,3,res,res] Long
    eval_labels = torch.LongTensor(eval_labels)  # [N] Long
    torch.save([eval_imgs, eval_labels], osp.join(out_dir, "eval.pt"))

    print(f"Saved to {out_dir}/train.pt and {out_dir}/eval.pt")
