from pathlib import Path
import imageio.v2 as imageio
import imgaug.augmenters as iaa
from tqdm import tqdm
import cv2
import numpy as np

DST = Path("../dataset/augmentation")
droot =  Path("../source")

class Rotator():
    def __init__(self, src_shape=(763,463), d:int=90):
        self.s = src_shape
        self.d = d
        self.r = None
        if d%90 == 0:
            self.r = iaa.Rot90(k=d//90)
        else:
            self.r = iaa.Affine(rotate=d, fit_output=True)

    def __call__(self, image:np.ndarray) -> np.ndarray:
        aug = self.r(image=image)
        if aug.shape[0] == self.s[0] and aug.shape[1] == self.s[1]:
            print("n * pi/2")
            return aug
        return cv2.resize(aug, self.s[::-1])

method_table = {
    
    "hf":iaa.Fliplr(1.0),
    "vf":iaa.Flipud(1.0),
    
    "r90":iaa.Rot90(k=1),
    "r180":iaa.Rot90(k=2),
    "r270":iaa.Rot90(k=3),
    
    "r20":Rotator(d=20),
    "rn20":Rotator(d=-20),
    "r40":Rotator(d=40),
    "rn40":Rotator(d=-40),
    "r60":Rotator(d=60),
    "rn60":Rotator(d=-60),

    "rs10": iaa.TranslateX(px=10),
    "ls10": iaa.TranslateX(px=-10),
    "ts10":iaa.TranslateY(px=-10),
    "ds10":iaa.TranslateY(px=10),

    "ls5": iaa.TranslateX(px=-5),
    "ts5":iaa.TranslateY(px=-5),
    "rs5": iaa.TranslateX(px=5),
    "ds5":iaa.TranslateY(px=5),
    
    "con":iaa.LinearContrast((1.6, 1.6)),
    "gb" :iaa.GaussianBlur(sigma=(1, 1))
}


not_use = {
    "Type0":["ts5", "rs5", "ls5", "ds5"],
    "Type1":["rn20", "r20", "rn40", "r40", "rn60", "r60",
             "ts10", "rs10", "ls10", "ds10"],
    "Type2":["ts5", "rs5", "ls5", "ds5"],
    "Type3":["ts5", "rs5", "ls5", "ds5"]
}

if __name__ == "__main__":
    table = {}

    for ti in droot.iterdir():
        t = ti.parts[-1]
        t_dst = DST/t
        t_dst.mkdir(parents=True, exist_ok=True)
        print(t_dst)
        if ti.is_dir():
            org_img = [_ for _ in ti.glob("*.jpg")]
            M = [k for k in method_table.keys() if k not in not_use[t]]
            print(t, M)
            for impath in tqdm(org_img):
                src = imageio.imread(impath)
                for mi in M:
                    aug_mi = method_table[mi](image=src)
                    imageio.imwrite(t_dst/f"aug_{mi}_{impath.stem}.jpg", aug_mi)
