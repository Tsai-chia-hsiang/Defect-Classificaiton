from pathlib import Path
import imageio.v2 as imageio
import imgaug.augmenters as iaa
from tqdm import tqdm
import os
import shutil

DST = Path("/mnt/Nami/bigdata2024/augmentation")
droot=  Path("dataset")/"p2"

method_table = {
    "hf":iaa.Sequential([iaa.Fliplr(1.0)]),
    "vf":iaa.Sequential([iaa.Flipud(1.0)]),
    "r90":iaa.Sequential([iaa.Rot90(k=1)]),
    "r180":iaa.Sequential([iaa.Rot90(k=2)]),
    "r270":iaa.Sequential([iaa.Rot90(k=3)]),
    "r20":iaa.Sequential([iaa.Affine(rotate=20)]),
    "rn20":iaa.Sequential([iaa.Affine(rotate=-20)]),
    "rs3": iaa.Sequential([iaa.TranslateX(px=3)]),
    "ls3": iaa.Sequential([iaa.TranslateX(px=-3)]),
    "ts3":iaa.Sequential([iaa.TranslateY(px=-3)]),
    "ds3":iaa.Sequential([iaa.TranslateY(px=3)]),
    "con":iaa.Sequential([iaa.LinearContrast((1.25, 1.4))]),
}
not_use = {
    "Type0":["con", "rs3"],
    "Type1":["rn20", "r20"],
    "Type2":["rs3", "ls3"],
    "Type3":["rs3", "ls3"]
}
if __name__ == "__main__":
    table = {}

    for ti in droot.iterdir():
        t = ti.parts[-1]
        t_dst = DST/t
        t_dst.mkdir(parents=True, exist_ok=True)
        print(t_dst)
        if ti.is_dir():
            org_img = [_ for _ in ti.glob("*.jpg") if 'aug' not in _.stem]
            M = [k for k in method_table.keys() if k not in not_use[t]]
            print(t, M)
            for impath in tqdm(org_img):
                src = imageio.imread(impath)
                shutil.copy(impath,t_dst/f"{impath.stem}.jpg")
                for mi in M:
                    aug_mi = method_table[mi](image=src)
                    imageio.imwrite(t_dst/f"aug_{mi}_{impath.stem}.jpg", aug_mi)
