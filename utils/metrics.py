import numpy as np
from tqdm import tqdm
from utils.render import linear_to_srgb
from typing import Tuple, List, Callable

def metric(
    gt: np.ndarray,
    imgs: List[np.ndarray],
    func: Callable,
    aggregate: Callable=np.mean,
    use_srgb: float=True,
) -> Tuple[List[float], List[np.ndarray]]:
    met = []
    agg_met = []
    if use_srgb:
        gt_ = linear_to_srgb(gt)
    else:
        gt_ = gt
    for img in tqdm(imgs):
        if use_srgb:
            img_ = linear_to_srgb(img)
        else:
            img_ = img
        met.append(func(img_, gt_))
        agg_met.append(aggregate(met[-1]))

    return agg_met, met

def se(
    gt: np.ndarray,
    imgs: List[np.ndarray],
    use_srgb: float=True
) -> Tuple[List[float], List[np.ndarray]]:
    return metric(gt, imgs, lambda x,y: np.square(x-y), use_srgb=use_srgb)

def ae(
    gt: np.ndarray,
    imgs: List[np.ndarray],
    use_srgb: float=True
) -> Tuple[List[float], List[np.ndarray]]:
    return metric(gt, imgs, lambda x,y: np.abs(x-y), use_srgb=use_srgb)

def l1(
    gt: np.ndarray,
    imgs: List[np.ndarray],
    use_srgb: float=True
) -> Tuple[List[float], List[np.ndarray]]:
    return metric(gt, imgs, lambda x,y: x-y, use_srgb=use_srgb)

def rae(
    gt: np.ndarray,
    imgs: List[np.ndarray],
    use_srgb: float=True
) -> Tuple[List[float], List[np.ndarray]]:
    return metric(gt, imgs, lambda x,y: np.abs(x-y) / (y+0.01), use_srgb=use_srgb)

def rse(
    gt: np.ndarray,
    imgs: List[np.ndarray],
    use_srgb: float=True
) -> Tuple[List[float], List[np.ndarray]]:
    return metric(gt, imgs, lambda x,y: np.square(x-y) / (np.square(y)+0.01), use_srgb=use_srgb)

def flip(
    gt: np.ndarray,
    imgs: List[np.ndarray],
    use_srgb: float=True
) -> Tuple[List[float], List[np.ndarray]]:
    raise NotImplementedError