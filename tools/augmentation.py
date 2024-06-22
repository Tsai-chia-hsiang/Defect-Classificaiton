import imageio.v2 as imageio
import imgaug.augmenters as iaa
from tqdm import tqdm
from pathlib import Path
import os

method_table = {
    "aughv":iaa.Sequential([iaa.Fliplr(1.0), iaa.Flipud(1.0)]),
    "augh":iaa.Sequential([iaa.Fliplr(1.0)]),
    "augv":iaa.Sequential([iaa.Flipud(1.0)]),
}
def remove_all_aug(root:Path):
    rm_img_path = [_ for _ in root.glob("*.jpg") if 'aug' in str(_)]
    for i in rm_img_path:
        os.remove(i)
    

if __name__ == "__main__":

    root = Path("../dataset")/"p2"
    aug_methods = [
        ["Type1", ["aughv", "augh", "augv"]],
        ["Type2", ["augh", "augv"]],
        ["Type3", ["aughv"]]
    ]
    for ti, aug_i in aug_methods:
        """remove_all_aug(root/ti)
        continue"""
        print(ti, aug_i)
        ti_img_path = [_ for _ in (root/ti).glob("*.jpg")]
        for to_aug in tqdm(ti_img_path):
            img_i = imageio.imread(to_aug)
            for mi in aug_i:
                dst = to_aug.parent/f"{mi}_{to_aug.stem}.jpg"
                aug_img_i = method_table[mi](image=img_i)
                imageio.imwrite(dst, aug_img_i)

    """
    for k, v in aug_methods.items():
        print(k)
        for i in v['to_aug']:
            print(i)
        print("+"*20)
    """