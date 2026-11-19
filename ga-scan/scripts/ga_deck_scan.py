#!/usr/bin/env python3
"""卡组扫盘 GA：固定 gen9 冠军旋钮（干净对照），GA 搜索最强的 6 张卡组合。

基因 = 6 个卡位（速/耐/力/智/智/友），每维从该类型 SSR 卡池里选一张。
适应度 = 搜索档(--search-n 128) 的 PT 均值，objective=pt。
多轮独立运行（不同种子段）防止少数偶发事件——同一卡组在多轮里都强才算真强。

用法：
  python3 ga_deck_scan.py --out DIR --games 40 --gens 12 --lam 8 --rounds 3
"""
from __future__ import annotations
import argparse, csv, json, math, os, subprocess, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np

ROOT = Path(__file__).resolve().parent.parent          # matsuri 仓根
DATA = ROOT / "data"
_DATA_ROOT = None  # main() 里由 --data-root 设置；云端跑时指向打包的 data 目录
EXE_DEFAULT = ROOT.parent / "target_v017" / "release" / "cardtest"
BASE_CONFIG = DATA / "matsuri_srch_sch8.toml"
CHAMP_SNIPPET = DATA / "champ_gen9.toml"               # gen9 冠军旋钮片段

# 6 个卡位：(名字, cardType, idrank=card_id*10+4)
# 当前双智基线: 速303114 耐303244 力303224 智303194 智302894 友303234
SLOT_TYPES = [0, 1, 2, 4, 4, 5]
BASELINE_IDRANK = [303114, 303244, 303224, 303194, 302894, 303234]  # 双智基线(idrank)
SELF_CHARA = None   # 本体角色 charaId，运行时由 --self-chara 指定（默认特别周1001）

# 种子段：每轮一个，防偶发
ROUND_SEEDS = [920000, 921000, 922000, 923000, 924000]


def load_pool(card_db: dict, self_chara: int) -> list[list[dict]]:
    """按 6 个卡位分别构造 SSR 满破候选池（剔除本体角色卡）。"""
    by_type: dict[int, list[dict]] = {}
    for cid, v in card_db.items():
        if v.get("cardType") not in SLOT_TYPES or v.get("rarity") != 3:
            continue
        if self_chara is not None and v.get("charaId") == self_chara:
            continue
        by_type.setdefault(v["cardType"], []).append({
            "card_id": int(cid),
            "idrank": int(cid) * 10 + 4,
            "name": v.get("cardName", cid),
            "chara": v.get("charaId"),
            "type": v["cardType"],
        })
    pools = []
    for t in SLOT_TYPES:
        ps = sorted(by_type.get(t, []), key=lambda x: x["card_id"])
        if t == 5:  # 友人槽：本剧本仅支持 SSR 吾妻(30323)，固定不可搜
            ps = [c for c in ps if c["card_id"] == 30323]
        if not ps:
            sys.exit(f"卡位 type={t} 没有候选卡")
        pools.append(ps)
    return pools


class Deck:
    """idx = 每个卡位在池内的索引（GA 基因）；内部映射成 idrank。"""

    def __init__(self, idx: list[int], pool: list[list[dict]]):
        self.idx = list(idx)
        self.cards = [pool[i][j] for i, j in enumerate(idx)]
        self.ids = [c["idrank"] for c in self.cards]
        comp = [0] * 5  # 构成码只含训练位5类(速耐力根智)，友人/团体不计
        for c in self.cards[:5]:
            t = c.get("type")
            if isinstance(t, int) and 0 <= t < 5:
                comp[t] += 1
        self.comp = comp
        self.code = "".join(map(str, comp))

    @property
    def names(self):
        return [c["name"] for c in self.cards]

    def to_json(self):
        return {"ids": self.ids, "names": self.names, "comp": self.comp,
                "cards_csv": ",".join(map(str, self.ids))}


def make_cfg(out: Path) -> Path:
    cfg = out / "scan_cfg.toml"
    cfg.write_text(BASE_CONFIG.read_text(encoding="utf-8")
                   + CHAMP_SNIPPET.read_text(encoding="utf-8"), encoding="utf-8")
    return cfg


def evaluate(deck: Deck, games: int, seed: int, cfg: Path, exe: Path,
             threads: int, out: Path, tag: str) -> dict:
    decks_csv = out / f"{tag}_decks.csv"
    decks_csv.write_text(
        "name,comp,pick,slot0,slot1,slot2,slot3,slot4,friend\n"
        f"scan,{deck.code},ga,{','.join(map(str, deck.ids[:5]))},{deck.ids[5]}\n",
        encoding="utf-8")
    csv_path = out / f"{tag}.csv"
    cmd = [str(exe), "run", "--decks", str(decks_csv), "--matsuri-config", str(cfg),
           "--games", str(games), "--seed", str(seed), "--threads", str(threads),
           "--search-n", "128", "--out", str(csv_path)]
    if _DATA_ROOT:
        cmd += ["--data-root", str(_DATA_ROOT)]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                          encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"cardtest 失败：\n{proc.stdout}\n{proc.stderr}")
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    def col(k):
        return float(np.mean([float(r[k]) for r in rows]))
    return {"pt": col("pt_mean"), "score": col("score_mean"),
            "friend_full": col("friend_full"),
            "five": [col(f"five{i}") for i in range(5)], "n": len(rows)}


def fitness_of(metrics: dict, objective: str, friend_penalty: float) -> float:
    key = "pt" if objective == "pt" else "score"
    return metrics[key] - friend_penalty * (1 - metrics["friend_full"])


def _conflict_slots(idx, pool) -> set[int]:
    """返回编成冲突的卡位集合：同角色(chara_id)或同卡(idrank)重复。
    引擎规则：存在重复人物(chara_id)即拒；两张智卡同卡也拒。chara=None 不参与角色判重。"""
    seen_chara, seen_idrank = {}, {}
    conflict = set()
    for i, j in enumerate(idx):
        c = pool[i][j]
        ir = c["idrank"]
        if ir in seen_idrank:
            conflict.add(i); conflict.add(seen_idrank[ir])
        else:
            seen_idrank[ir] = i
        ch = c.get("chara")
        if ch is not None:
            if ch in seen_chara:
                conflict.add(i); conflict.add(seen_chara[ch])
            else:
                seen_chara[ch] = i
    return conflict


def _fix_conflicts(idx, pool, rng) -> list[int]:
    """迭代重抽冲突卡位直到编成合法；友人槽(5)固定(仅1张)不动。"""
    ids = list(idx)
    for _ in range(100):
        bad = _conflict_slots(ids, pool)
        if not bad:
            return ids
        progressed = False
        for i in sorted(bad, reverse=True):
            if i == 5:  # 友人固定，跳过（其冲突方是非友人槽，会被重抽）
                continue
            if len(pool[i]) <= 1:
                continue
            ids[i] = int(rng.integers(len(pool[i])))
            progressed = True
        if not progressed:
            # 全部冲突槽都不可动（极端），重置非友人槽
            for i in range(5):
                if len(pool[i]) > 1:
                    ids[i] = int(rng.integers(len(pool[i])))
    return ids


def random_deck(pool, rng) -> list[int]:
    ids = [int(rng.integers(len(p))) for p in pool]
    return _fix_conflicts(ids, pool, rng)


def mutate(ids, pool, rng, pmut):
    ids = list(ids)
    for i in range(len(ids)):
        if rng.random() < pmut:
            ids[i] = int(rng.integers(len(pool[i])))
    return _fix_conflicts(ids, pool, rng)


def crossover(a, b, rng, pool):
    mask = rng.random(len(a)) < 0.5
    ids = [x if m else y for x, y, m in zip(a, b, mask)]
    return _fix_conflicts(ids, pool, rng)


def log(path: Path, obj):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def baseline_idx(pool) -> list[int] | None:
    """双智基线的池内索引（若某卡被剔除则返回 None）。"""
    idx = []
    for i, idr in enumerate(BASELINE_IDRANK):
        hit = [j for j, c in enumerate(pool[i]) if c["idrank"] == idr]
        if not hit:
            return None
        idx.append(hit[0])
    return idx


def run_round(args, rnd: int, seed: int, pool, cfg, exe, out) -> dict:
    rng = np.random.default_rng(args.rng_seed + rnd * 1000)
    tagdir = out / f"round{rnd}"
    tagdir.mkdir(exist_ok=True)
    bidx = baseline_idx(pool)
    # 初始种群 = 基线 + 随机（基因均为池内索引）
    pop = [bidx[:]] if bidx is not None else []
    while len(pop) < args.lam:
        pop.append(random_deck(pool, rng))
    gens_path = tagdir / "gens.jsonl"
    best_ids, best_fit = None, -1e18
    t0 = time.time()
    for gen in range(args.gens):
        # 评估全部个体（同种子 CRN）
        results = []
        for i, ids in enumerate(pop):
            deck = Deck(ids, pool)
            m = evaluate(deck, args.games, seed + gen * args.games, cfg, exe,
                         args.threads, tagdir, f"g{gen:02d}_i{i:02d}")
            results.append((fitness_of(m, args.objective, args.friend_penalty), ids, m))
        results.sort(key=lambda x: -x[0])
        gen_best = results[0]
        if gen_best[0] > best_fit:
            best_fit, best_ids = gen_best[0], gen_best[1][:]
        log(gens_path, {"round": rnd, "gen": gen, "best_fit": gen_best[0],
                        "best_pt": gen_best[2]["pt"],
                        "best_ids": Deck(gen_best[1], pool).ids,
                        "best_names": Deck(gen_best[1], pool).names,
                        "median_fit": float(np.median([r[0] for r in results])),
                        "secs": round(time.time() - t0, 1)})
        print(f"r{rnd} gen{gen:2d}  best {gen_best[0]:9.1f} (PT {gen_best[2]['pt']:8.1f})  "
              f"median {np.median([r[0] for r in results]):9.1f}  "
              f"{time.time()-t0:6.0f}s", flush=True)
        # 精英保留 + 锦标赛 + 交叉变异
        elite = [x[1][:] for x in results[:2]]
        newpop = elite[:]
        while len(newpop) < args.lam:
            a = results[int(rng.integers(0, len(results) // 2))][1]
            b = results[int(rng.integers(0, len(results) // 2))][1]
            newpop.append(mutate(crossover(a, b, rng, pool), pool, rng, args.pmut))
        pop = newpop
    # 该轮冠军用大样本复核（防小样本虚高）
    champ = Deck(best_ids, pool)
    verify = evaluate(champ, args.verify_games, seed + 500000, cfg, exe,
                      args.threads, tagdir, "verify_best")
    baseline = Deck(bidx, pool) if bidx is not None else None
    base_v = evaluate(baseline, args.verify_games, seed + 500000, cfg, exe,
                      args.threads, tagdir, "verify_base") if baseline else None
    result = {"round": rnd, "seed": seed, "best": champ.to_json(),
              "best_fit": best_fit, "verify": verify,
              "baseline": baseline.to_json() if baseline else None,
              "baseline_verify": base_v}
    (tagdir / "best.json").write_text(json.dumps(result, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--exe", type=Path, default=EXE_DEFAULT)
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--verify-games", type=int, default=200)
    ap.add_argument("--gens", type=int, default=12)
    ap.add_argument("--lam", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--rng-seed", type=int, default=12345)
    ap.add_argument("--objective", choices=["pt", "score"], default="pt")
    ap.add_argument("--friend-penalty", type=float, default=500.0)
    ap.add_argument("--self-chara", type=int, default=1001, help="本体马娘charaId(特别周=1001)")
    ap.add_argument("--pmut", type=float, default=0.25)
    ap.add_argument("--seeds", type=int, nargs="+", default=ROUND_SEEDS)
    ap.add_argument("--data-root", type=Path, default=DATA,
                    help="二进制读取的数据根；默认脚本同级data，云端可显式指定")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    global _DATA_ROOT
    _DATA_ROOT = args.data_root

    card_db = json.load((DATA / "cardDB.json").open(encoding="utf-8"))
    pool = load_pool(card_db, args.self_chara)
    cfg = make_cfg(args.out)
    print(f"# 卡组扫盘GA 目标={args.objective} 卡位={SLOT_TYPES} "
          f"池大小={[len(p) for p in pool]} 轮数={args.rounds} 每轮gens={args.gens} "
          f"lam={args.lam} games={args.games} verify={args.verify_games}", flush=True)
    print(f"# 各池卡数：", {t: len(p) for t, p in zip(SLOT_TYPES, pool)}, flush=True)

    seeds = args.seeds[:args.rounds]
    all_results = []
    for rnd, seed in enumerate(seeds):
        res = run_round(args, rnd, seed, pool, cfg, args.exe, args.out)
        all_results.append(res)
        print(f"=== r{rnd} 复核 PT={res['verify']['pt']:.1f} "
              f"vs 基线 PT={res['baseline_verify']['pt'] if res['baseline_verify'] else 'NA'} "
              f"冠军={'|'.join(res['best']['names'])} ===", flush=True)
        log(args.out / "rounds.jsonl", res)

    # 汇总：多轮冠军里选复核 PT 最高的
    champs = sorted(all_results, key=lambda r: -r["verify"]["pt"])
    (args.out / "summary.json").write_text(json.dumps(
        {"objective": args.objective, "rounds": len(all_results),
         "winner": champs[0], "all": all_results}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    w = champs[0]
    print(f"\n### 扫盘冠军（{len(all_results)}轮中最高复核PT）", flush=True)
    print(f"PT={w['verify']['pt']:.1f} 评分={w['verify']['score']:.1f} "
          f"卡={'|'.join(w['best']['names'])}", flush=True)
    print(f"完整结果：{args.out/'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
