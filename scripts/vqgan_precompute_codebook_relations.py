from __future__ import annotations
import argparse
import os
import sys
from typing import Tuple

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.append(".")
from taming.models.vqgan import VQModel


ALIVE_TOKEN_IDS = np.array(
    [    23,    66,    81,    91,   112,   125,   128,   129,   134,
         142,   153,   167,   194,   217,   245,   271,   351,   358,
         381,   424,   432,   468,   572,   598,   601,   632,   633,
         637,   641,   652,   663,   689,   699,   752,   754,   774,
         811,   825,   852,   859,   862,   867,   882,   903,   910,
         920,   926,   941,   948,   972,   981,  1008,  1024,  1052,
        1069,  1125,  1126,  1133,  1139,  1183,  1193,  1208,  1227,
        1250,  1258,  1261,  1268,  1270,  1274,  1291,  1304,  1330,
        1344,  1363,  1400,  1402,  1464,  1476,  1495,  1497,  1523,
        1536,  1553,  1556,  1567,  1570,  1571,  1586,  1597,  1598,
        1618,  1627,  1632,  1635,  1663,  1702,  1704,  1711,  1721,
        1731,  1736,  1742,  1755,  1765,  1810,  1811,  1847,  1850,
        1872,  1905,  1933,  1936,  1941,  1948,  1949,  1952,  1962,
        1969,  1972,  1989,  1994,  2029,  2032,  2036,  2061,  2063,
        2076,  2091,  2102,  2105,  2120,  2127,  2129,  2157,  2193,
        2214,  2216,  2221,  2237,  2255,  2273,  2279,  2325,  2334,
        2339,  2383,  2384,  2425,  2426,  2428,  2439,  2443,  2469,
        2471,  2474,  2481,  2486,  2514,  2519,  2523,  2524,  2528,
        2572,  2594,  2595,  2601,  2635,  2637,  2649,  2652,  2683,
        2701,  2705,  2737,  2778,  2779,  2810,  2813,  2872,  2878,
        2881,  2891,  2895,  2920,  2922,  2953,  2959,  2987,  3001,
        3006,  3029,  3037,  3042,  3046,  3052,  3061,  3103,  3132,
        3135,  3166,  3168,  3170,  3210,  3212,  3262,  3272,  3283,
        3303,  3342,  3379,  3386,  3402,  3424,  3442,  3516,  3540,
        3547,  3569,  3583,  3613,  3648,  3708,  3710,  3742,  3768,
        3771,  3801,  3804,  3809,  3811,  3812,  3821,  3823,  3846,
        3851,  3863,  3876,  3915,  3927,  3929,  3932,  3957,  3963,
        3974,  4011,  4039,  4054,  4060,  4075,  4099,  4135,  4147,
        4158,  4159,  4179,  4198,  4203,  4236,  4238,  4243,  4265,
        4297,  4301,  4319,  4326,  4331,  4395,  4460,  4465,  4473,
        4483,  4493,  4506,  4590,  4629,  4635,  4642,  4644,  4648,
        4652,  4682,  4697,  4704,  4710,  4720,  4751,  4762,  4771,
        4811,  4815,  4816,  4817,  4830,  4839,  4868,  4869,  4903,
        4904,  4912,  4942,  4945,  4950,  4967,  5031,  5053,  5062,
        5065,  5068,  5071,  5093,  5098,  5107,  5113,  5171,  5192,
        5201,  5213,  5238,  5243,  5245,  5286,  5288,  5290,  5305,
        5306,  5314,  5319,  5360,  5370,  5401,  5410,  5416,  5418,
        5452,  5495,  5509,  5511,  5522,  5537,  5542,  5564,  5566,
        5588,  5605,  5625,  5644,  5651,  5657,  5662,  5666,  5673,
        5688,  5693,  5722,  5744,  5756,  5762,  5772,  5791,  5827,
        5839,  5848,  5852,  5885,  5900,  5908,  5918,  5919,  5960,
        5976,  5984,  6001,  6007,  6018,  6058,  6059,  6060,  6074,
        6078,  6092,  6113,  6123,  6172,  6204,  6242,  6243,  6271,
        6274,  6328,  6331,  6348,  6358,  6359,  6362,  6369,  6386,
        6442,  6464,  6467,  6477,  6483,  6522,  6523,  6524,  6534,
        6535,  6598,  6601,  6605,  6606,  6628,  6660,  6690,  6747,
        6756,  6764,  6779,  6783,  6790,  6805,  6808,  6876,  6879,
        6882,  6883,  6889,  6920,  6930,  6933,  6945,  6948,  6965,
        6992,  6994,  7020,  7042,  7043,  7054,  7080,  7103,  7106,
        7151,  7160,  7163,  7191,  7226,  7236,  7259,  7266,  7292,
        7315,  7365,  7372,  7398,  7407,  7424,  7436,  7465,  7466,
        7467,  7482,  7491,  7504,  7516,  7540,  7542,  7575,  7576,
        7580,  7634,  7656,  7664,  7689,  7696,  7701,  7712,  7716,
        7721,  7726,  7738,  7749,  7751,  7752,  7771,  7781,  7798,
        7800,  7806,  7837,  7841,  7864,  7896,  7900,  7927,  7972,
        7979,  7986,  7996,  8026,  8075,  8111,  8131,  8139,  8144,
        8163,  8170,  8173,  8202,  8207,  8222,  8225,  8228,  8269,
        8276,  8283,  8302,  8322,  8328,  8361,  8380,  8382,  8408,
        8412,  8413,  8431,  8447,  8485,  8493,  8517,  8527,  8566,
        8574,  8581,  8583,  8596,  8609,  8612,  8646,  8663,  8668,
        8694,  8698,  8717,  8736,  8737,  8770,  8783,  8819,  8834,
        8862,  8867,  8877,  8888,  8928,  8946,  8983,  9000,  9010,
        9019,  9032,  9035,  9043,  9103,  9117,  9131,  9136,  9150,
        9157,  9183,  9186,  9196,  9203,  9206,  9212,  9213,  9255,
        9287,  9296,  9325,  9345,  9346,  9354,  9355,  9377,  9381,
        9390,  9393,  9407,  9451,  9453,  9470,  9479,  9488,  9501,
        9502,  9514,  9534,  9570,  9589,  9598,  9630,  9650,  9656,
        9673,  9710,  9713,  9720,  9753,  9756,  9819,  9834,  9837,
        9889,  9890,  9897,  9899,  9900,  9905,  9922,  9957,  9979,
       10019, 10020, 10037, 10042, 10075, 10081, 10089, 10120, 10121,
       10171, 10180, 10200, 10216, 10225, 10227, 10228, 10274, 10278,
       10280, 10295, 10297, 10319, 10362, 10397, 10407, 10473, 10487,
       10508, 10529, 10543, 10549, 10553, 10561, 10620, 10623, 10627,
       10629, 10639, 10689, 10710, 10721, 10741, 10769, 10797, 10802,
       10805, 10813, 10845, 10858, 10890, 10895, 10909, 10912, 10956,
       10991, 10997, 11004, 11012, 11024, 11053, 11075, 11096, 11110,
       11147, 11196, 11210, 11211, 11227, 11228, 11235, 11272, 11298,
       11306, 11307, 11325, 11326, 11327, 11334, 11342, 11348, 11370,
       11388, 11389, 11427, 11429, 11456, 11462, 11466, 11480, 11492,
       11497, 11500, 11574, 11587, 11591, 11594, 11605, 11610, 11656,
       11665, 11674, 11690, 11700, 11708, 11712, 11715, 11716, 11718,
       11725, 11738, 11792, 11811, 11820, 11904, 11918, 11927, 11941,
       11951, 11993, 12016, 12025, 12026, 12044, 12062, 12082, 12124,
       12139, 12142, 12143, 12145, 12159, 12165, 12167, 12183, 12187,
       12215, 12272, 12281, 12285, 12290, 12348, 12353, 12362, 12426,
       12480, 12509, 12513, 12541, 12557, 12566, 12607, 12618, 12623,
       12625, 12627, 12659, 12672, 12722, 12786, 12823, 12838, 12842,
       12857, 12861, 12892, 12927, 12951, 13010, 13037, 13050, 13094,
       13107, 13157, 13164, 13175, 13183, 13226, 13236, 13291, 13293,
       13326, 13342, 13357, 13370, 13401, 13417, 13427, 13440, 13500,
       13512, 13513, 13519, 13543, 13579, 13586, 13614, 13670, 13686,
       13687, 13712, 13757, 13769, 13790, 13810, 13826, 13845, 13871,
       13884, 13909, 13967, 13989, 14008, 14037, 14059, 14068, 14076,
       14102, 14129, 14134, 14140, 14142, 14151, 14164, 14176, 14185,
       14225, 14240, 14241, 14246, 14247, 14248, 14257, 14283, 14319,
       14379, 14416, 14420, 14435, 14447, 14450, 14467, 14473, 14476,
       14494, 14531, 14547, 14592, 14602, 14633, 14673, 14681, 14682,
       14703, 14712, 14717, 14747, 14782, 14785, 14793, 14839, 14843,
       14849, 14853, 14878, 14899, 14931, 14933, 14965, 14975, 15047,
       15076, 15080, 15089, 15111, 15149, 15180, 15186, 15189, 15213,
       15249, 15253, 15265, 15276, 15282, 15303, 15331, 15342, 15366,
       15383, 15387, 15421, 15422, 15430, 15431, 15433, 15464, 15514,
       15523, 15525, 15528, 15558, 15565, 15575, 15618, 15630, 15631,
       15645, 15647, 15664, 15695, 15702, 15703, 15729, 15739, 15771,
       15787, 15798, 15813, 15826, 15842, 15881, 15883, 15891, 15910,
       15916, 15917, 15936, 15945, 15947, 15963, 15978, 16017, 16018,
       16021, 16031, 16035, 16068, 16146, 16147, 16160, 16223, 16226,
       16242, 16248, 16274, 16279, 16287, 16301, 16314, 16338, 16374]
    , dtype=np.int32)


def load_model(config_path: str, ckpt_path: str, device: torch.device) -> VQModel:
    config = OmegaConf.load(config_path)
    model = VQModel(**config.model.params)
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return model


def compute_maps(
    codebook: torch.Tensor,
    alive_token_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if alive_token_ids.ndim != 1:
        raise ValueError("alive_token_ids must be a 1D array")

    n_embed, _ = codebook.shape
    alive_token_ids = alive_token_ids.astype(np.int64)
    if alive_token_ids.size < 2:
        raise ValueError("alive_token_ids must contain at least 2 tokens")
    if np.any(alive_token_ids < 0) or np.any(alive_token_ids >= n_embed):
        raise ValueError("alive_token_ids contain out-of-range values")
    if len(np.unique(alive_token_ids)) != len(alive_token_ids):
        raise ValueError("alive_token_ids contain duplicates")

    device = codebook.device
    alive_t = torch.from_numpy(alive_token_ids).to(device=device, dtype=torch.long)
    e = codebook.index_select(0, alive_t)

    norms = (e * e).sum(dim=1, keepdim=True)
    dist2 = norms + norms.transpose(0, 1) - 2.0 * (e @ e.transpose(0, 1))
    dist2 = dist2.clamp_min_(0.0)

    dist2_min = dist2.clone()
    dist2_min.fill_diagonal_(float("inf"))
    min_j = dist2_min.argmin(dim=1)

    dist2_max = dist2.clone()
    dist2_max.fill_diagonal_(-float("inf"))
    max_j = dist2_max.argmax(dim=1)

    eps = 1e-12
    e_norm = e / (e.norm(dim=1, keepdim=True) + eps)
    cos = e_norm @ e_norm.transpose(0, 1)
    abs_cos = cos.abs()
    abs_cos.fill_diagonal_(float("inf"))
    ortho_j = abs_cos.argmin(dim=1)

    alive_np = alive_token_ids.astype(np.int32)
    min_j_np = min_j.detach().cpu().numpy().astype(np.int64)
    max_j_np = max_j.detach().cpu().numpy().astype(np.int64)
    ortho_j_np = ortho_j.detach().cpu().numpy().astype(np.int64)

    min_dist_idx = np.full((n_embed,), -1, dtype=np.int32)
    max_dist_idx = np.full((n_embed,), -1, dtype=np.int32)
    ortho_idx = np.full((n_embed,), -1, dtype=np.int32)

    min_dist_idx[alive_np] = alive_np[min_j_np]
    max_dist_idx[alive_np] = alive_np[max_j_np]
    ortho_idx[alive_np] = alive_np[ortho_j_np]

    if np.any(min_dist_idx[alive_np] == alive_np):
        raise RuntimeError("min_dist_idx contains self-maps")
    if np.any(max_dist_idx[alive_np] == alive_np):
        raise RuntimeError("max_dist_idx contains self-maps")
    if np.any(ortho_idx[alive_np] == alive_np):
        raise RuntimeError("ortho_idx contains self-maps")

    return min_dist_idx, max_dist_idx, ortho_idx


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        default="/checkpoints/vqgan_imagenet_f16_16384/model.yaml",
        type=str,
    )
    p.add_argument(
        "--ckpt",
        default="/checkpoints/vqgan_imagenet_f16_16384/last.ckpt",
        type=str,
    )
    p.add_argument(
        "--out",
        default="vqgan_codebook_relations.npz",
        type=str,
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        type=str,
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model = load_model(args.config, args.ckpt, device)
    codebook = model.quantize.embedding.weight.detach()

    min_dist_idx, max_dist_idx, ortho_idx = compute_maps(codebook, ALIVE_TOKEN_IDS)

    out_path = args.out
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    np.savez(
        out_path,
        alive_token_ids=ALIVE_TOKEN_IDS.astype(np.int32),
        min_dist_idx=min_dist_idx,
        max_dist_idx=max_dist_idx,
        ortho_idx=ortho_idx,
        n_embed=np.int32(codebook.shape[0]),
        embed_dim=np.int32(codebook.shape[1]),
    )

    alive_ct = int(ALIVE_TOKEN_IDS.shape[0])
    print(f"Saved: {out_path}")
    print(f"n_embed={int(codebook.shape[0])} embed_dim={int(codebook.shape[1])} alive={alive_ct}")


if __name__ == "__main__":
    main()

