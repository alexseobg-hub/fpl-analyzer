#!/usr/bin/env python3
"""FPL analyzer engine: fetches the public FPL API and writes docs/data.json.

Usage:  python engine/build.py            (config in engine/config.json)
Runs nightly in GitHub Actions; no API keys needed.
"""
import json, math, os, sys, time, ssl, urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from itertools import combinations

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = json.load(open(os.path.join(ROOT, "engine", "config.json"), encoding="utf-8"))
API = "https://fantasy.premierleague.com/api/"
HORIZON = CFG.get("horizon", 6)
WEIGHTS = [1.0, 0.92, 0.85, 0.78, 0.72, 0.66, 0.6, 0.55][:HORIZON]
CTX = ssl.create_default_context(cafile=os.environ.get("SSL_CERT_FILE") or None)


def get(path, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(API + path, headers={"User-Agent": "fpl-analyzer/1.0"})
            with urllib.request.urlopen(req, timeout=30, context=CTX) as r:
                return json.load(r)
        except Exception as e:  # noqa
            if i == tries - 1:
                print("FAIL", path, e, file=sys.stderr)
                return None
            time.sleep(1.5 * (i + 1))


def pmap(fn, items, workers=8):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(fn, items))


def poisson_floor_half(lam):
    """E[floor(G/2)] for G ~ Poisson(lam)."""
    s, p = 0.0, math.exp(-lam)
    for g in range(0, 12):
        if g > 0:
            p *= lam / g
        s += (g // 2) * p
    return s


# ------------------------------------------------------------------ fetch
bs = get("bootstrap-static/")
fixtures = get("fixtures/")
events = bs["events"]
cur_ev = next((e["id"] for e in events if e["is_current"]), 0)
nxt = next((e for e in events if e["is_next"]), None)
next_ev = nxt["id"] if nxt else cur_ev + 1
gws = [g for g in range(next_ev, next_ev + HORIZON) if g <= 38]
POS = {t["id"]: t["singular_name_short"] for t in bs["element_types"]}
TEAMS = {t["id"]: t for t in bs["teams"]}
SC = bs["game_config"]["scoring"]
els = {e["id"]: e for e in bs["elements"]}

active_ids = [i for i, e in els.items() if e["minutes"] > 0 and e["status"] != "u"]
summaries = dict(zip(active_ids, pmap(lambda i: get(f"element-summary/{i}/"), active_ids)))

# ------------------------------------------------------------------ team strength from per-match xG
fx_by_id = {f["id"]: f for f in fixtures}
team_xg = {}  # fixture -> team -> xG
for pid, s in summaries.items():
    if not s:
        continue
    t = els[pid]["team"]
    for h in s["history"]:
        team_xg.setdefault(h["fixture"], {}).setdefault(t, 0.0)
        team_xg[h["fixture"]][t] += float(h["expected_goals"])

played = [f for f in fixtures if f["finished"] and f["id"] in team_xg]
att = {t: [] for t in TEAMS}
dfn = {t: [] for t in TEAMS}
for f in sorted(played, key=lambda f: f["event"] or 0):
    h, a = f["team_h"], f["team_a"]
    xh, xa = team_xg[f["id"]].get(h, 0.0), team_xg[f["id"]].get(a, 0.0)
    att[h].append(xh); dfn[h].append(xa)
    att[a].append(xa); dfn[a].append(xh)

allx = [x for v in att.values() for x in v]
MU = sum(allx) / max(len(allx), 1)


def wavg(vals, prior, k=4.0, decay=0.93):
    n = len(vals)
    w = [decay ** (n - 1 - i) for i in range(n)]
    return (sum(v * wi for v, wi in zip(vals, w)) + prior * k) / (sum(w) + k)


ATT = {t: wavg(att[t], MU) for t in TEAMS}
DEF = {t: wavg(dfn[t], MU) for t in TEAMS}
HF = 1.08  # home multiplier on expected goals


def lam(team, opp, home):
    return MU * (ATT[team] / MU) * (DEF[opp] / MU) * (HF if home else 1 / HF)


# fixtures per team per GW (handles blanks/doubles)
team_fx = {t: {g: [] for g in gws} for t in TEAMS}
for f in fixtures:
    if f["event"] in gws:
        team_fx[f["team_h"]][f["event"]].append((f["team_a"], True))
        team_fx[f["team_a"]][f["event"]].append((f["team_h"], False))

# ------------------------------------------------------------------ player model
PRIOR_XG90 = {"GKP": 0.0, "DEF": 0.04, "MID": 0.13, "FWD": 0.32}
PRIOR_XA90 = {"GKP": 0.01, "DEF": 0.05, "MID": 0.12, "FWD": 0.1}
DC_THR = {"DEF": 10, "MID": 12, "FWD": 12}


def player_model(e):
    pos = POS[e["element_type"]]
    s = summaries.get(e["id"])
    hist = s["history"] if s else []
    last = hist[-5:]
    mins = e["minutes"]
    n90 = mins / 90.0
    xg90 = (float(e["expected_goals"]) + PRIOR_XG90[pos] * 3) / (n90 + 3)
    xa90 = (float(e["expected_assists"]) + PRIOR_XA90[pos] * 3) / (n90 + 3)
    if last:
        exp_min = sum(h["minutes"] for h in last) / len(last)
        p60 = sum(h["minutes"] >= 60 for h in last) / len(last)
        papp = sum(h["minutes"] > 0 for h in last) / len(last)
    else:
        exp_min = p60 = papp = 0.0
    # availability
    cop = e["chance_of_playing_next_round"]
    if e["status"] in ("u", "n"):
        avail = 0.0
    elif cop is not None:
        avail = cop / 100.0
    else:
        avail = 1.0 if e["status"] == "a" else 0.5
    # defensive contribution hit-rate in starts
    starts = [h for h in hist[-10:] if h["minutes"] >= 60]
    dc_rate = 0.0
    if pos in DC_THR and starts:
        dc_rate = sum(h["defensive_contribution"] >= DC_THR[pos] for h in starts) / len(starts)
    per90 = lambda v: (v / n90) if n90 >= 1 else 0.0
    bonus90 = per90(e["bonus"])
    saves90 = per90(e["saves"])
    yc90 = per90(e["yellow_cards"])
    team = e["team"]
    out = []
    for k, g in enumerate(gws):
        a = 1 - (1 - avail) * (0.45 ** k)  # recovery ramp for doubts/injuries
        tot = 0.0
        for opp, home in team_fx[team][g]:
            lf, la = lam(team, opp, home), lam(opp, team, not home)
            m = exp_min * a
            q60, qa = p60 * a, papp * a
            att_mult = lf / ATT[team]
            pts = qa + q60
            pts += xg90 * att_mult * m / 90 * SC["goals_scored"][pos]
            pts += xa90 * att_mult * m / 90 * SC["assists"]
            pts += q60 * math.exp(-la) * SC["clean_sheets"][pos]
            if pos in ("GKP", "DEF"):
                pts -= (m / 90) * poisson_floor_half(la)
            if pos == "GKP":
                pts += saves90 * (la / MU) * m / 90 / 3
            pts += dc_rate * q60 * 2
            pts += bonus90 * (att_mult ** 0.5) * m / 90
            pts -= yc90 * m / 90
            tot += pts
        out.append(round(tot, 2))
    form5 = [h["total_points"] for h in last]
    return {
        "xpts": out,
        "xmin": round(exp_min * avail, 1),
        "avail": round(avail, 2),
        "xg90": round(xg90, 3),
        "xa90": round(xa90, 3),
        "form5": form5,
        "dc_rate": round(dc_rate, 2),
    }


players = []
for e in els.values():
    if e["status"] == "u" and e["minutes"] == 0:
        continue
    pos = POS[e["element_type"]]
    pm = player_model(e) if e["id"] in summaries else {
        "xpts": [0.0] * len(gws), "xmin": 0, "avail": 1.0 if e["status"] == "a" else 0, "xg90": 0, "xa90": 0, "form5": [], "dc_rate": 0}
    players.append({
        "id": e["id"], "name": e["web_name"], "team": e["team"], "pos": pos,
        "price": e["now_cost"] / 10, "own": float(e["selected_by_percent"]),
        "pts": e["total_points"], "mins": e["minutes"], "starts": e["starts"],
        "g": e["goals_scored"], "a": e["assists"],
        "xg": float(e["expected_goals"]), "xa": float(e["expected_assists"]),
        "bonus": e["bonus"], "dc": e["defensive_contribution"],
        "form": float(e["form"]), "ep_next": float(e["ep_next"] or 0),
        "status": e["status"], "news": e["news"], "cop": e["chance_of_playing_next_round"],
        "tin": e["transfers_in_event"], "tout": e["transfers_out_event"],
        "pc_pct": float(e["price_change_percent"] or 0),
        "pc_proj": [float(p["projected_percent"]) for p in (e.get("price_change_projections") or [])],
        "cost_ev": e["cost_change_event"] / 10, "cost_start": e["cost_change_start"] / 10,
        "pen": e["penalties_order"], "fk": e["direct_freekicks_order"], "cor": e["corners_and_indirect_freekicks_order"],
        **pm,
        "xh": round(sum(x * w for x, w in zip(pm["xpts"], WEIGHTS)), 2),
    })
P = {p["id"]: p for p in players}

# ------------------------------------------------------------------ squad tools
def best_xi(ids, gi):
    """Return (xi_points_with_captain, xi_ids, captain_id, bench_sum) for GW index gi."""
    by = {"GKP": [], "DEF": [], "MID": [], "FWD": []}
    for i in ids:
        p = P.get(i)
        if p:
            by[p["pos"]].append((p["xpts"][gi], i))
    for v in by.values():
        v.sort(reverse=True)
    if not by["GKP"]:
        return 0, [], None, 0
    xi = [by["GKP"][0]]
    xi += by["DEF"][:3] + by["MID"][:2] + by["FWD"][:1]
    rest = sorted(by["DEF"][3:] + by["MID"][2:] + by["FWD"][1:], reverse=True)
    need = 11 - len(xi)
    xi += rest[:need]
    bench = rest[need:]
    # captain: small upside bonus for attackers (higher ceiling at equal xPts)
    cap = max(xi, key=lambda x: x[0] * (1.06 if P[x[1]]["pos"] in ("MID", "FWD") else 1.0))
    total = sum(x for x, _ in xi) + cap[0]
    return total, [i for _, i in xi], cap[1], sum(x for x, _ in bench)


def squad_value(ids):
    v = 0.0
    for gi, w in enumerate(WEIGHTS[: len(gws)]):
        t, _, _, b = best_xi(ids, gi)
        v += w * (t + 0.1 * b)
    return v


# ------------------------------------------------------------------ my team
def entry_bundle(eid, with_hist=True):
    ent = get(f"entry/{eid}/")
    picks = get(f"entry/{eid}/event/{cur_ev}/picks/") if cur_ev else None
    hist = get(f"entry/{eid}/history/") if with_hist else None
    return ent, picks, hist


def free_transfers(hist):
    chips = {c["event"]: c["name"] for c in hist["chips"]}
    ft = 0
    for row in hist["current"]:
        ev = row["event"]
        if ev == 1:
            ft = 1
            continue
        if chips.get(ev) in ("wildcard", "freehit"):
            continue  # banked FTs are kept, but no extra FT is gained for that GW
        ft = max(ft - row["event_transfers"], 0)
        ft = min(ft + 1, 5)
    return ft


def chips_left(hist):
    used = [(c["name"], c["event"]) for c in hist["chips"]]
    left = []
    for c in bs["chips"]:
        if not (c["start_event"] <= next_ev <= c["stop_event"]):
            continue  # only chips in the current half-season window
        if not any(n == c["name"] and c["start_event"] <= ev <= c["stop_event"] for n, ev in used):
            left.append({"name": c["name"], "until": c["stop_event"], "from": c["start_event"]})
    return left


# ------------------------------------------------------------------ fixture ticker
ticker = []
for t, tm in TEAMS.items():
    row = []
    for g in gws:
        row.append([{"opp": TEAMS[o]["short_name"], "h": h,
                     "att": round(lam(t, o, h), 2), "def": round(lam(o, t, not h), 2)} for o, h in team_fx[t][g]])
    ticker.append({"id": t, "short": tm["short_name"], "name": tm["name"], "fx": row,
                   "att": round(ATT[t], 2), "def": round(DEF[t], 2)})


os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
# ------------------------------------------------------------------ team ratings (0-100) vs optimal squad
def optimal_value(gis, budget):
    """Best possible xP (XI + captain + 0.1*bench) for a fixed 15-man squad over GW indexes gis."""
    try:
        import pulp
    except ImportError:
        return None
    pool = [p for p in players if p["avail"] > 0 or p["mins"] > 0]
    prob = pulp.LpProblem("opt", pulp.LpMaximize)
    x = {p["id"]: pulp.LpVariable(f"x{p['id']}", cat="Binary") for p in pool}
    y = {(p["id"], g): pulp.LpVariable(f"y{p['id']}_{g}", cat="Binary") for p in pool for g in gis}
    c = {(p["id"], g): pulp.LpVariable(f"c{p['id']}_{g}", cat="Binary") for p in pool for g in gis}
    prob += pulp.lpSum(p["xpts"][g] * (y[p["id"], g] + c[p["id"], g]) + 0.1 * p["xpts"][g] * (x[p["id"]] - y[p["id"], g])
                       for p in pool for g in gis)
    prob += pulp.lpSum(p["price"] * x[p["id"]] for p in pool) <= budget
    for pos, n in (("GKP", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)):
        prob += pulp.lpSum(x[p["id"]] for p in pool if p["pos"] == pos) == n
    for t in TEAMS:
        prob += pulp.lpSum(x[p["id"]] for p in pool if p["team"] == t) <= 3
    for g in gis:
        prob += pulp.lpSum(y[p["id"], g] for p in pool) == 11
        prob += pulp.lpSum(c[p["id"], g] for p in pool) == 1
        for pos, lo, hi in (("GKP", 1, 1), ("DEF", 3, 5), ("MID", 2, 5), ("FWD", 1, 3)):
            s = pulp.lpSum(y[p["id"], g] for p in pool if p["pos"] == pos)
            prob += s >= lo
            prob += s <= hi
        for p in pool:
            prob += y[p["id"], g] <= x[p["id"]]
            prob += c[p["id"], g] <= y[p["id"], g]
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=60) if pulp.PULP_CBC_CMD().available() else pulp.HiGHS(msg=False, timeLimit=60)
    prob.solve(solver)
    if pulp.LpStatus[prob.status] not in ("Optimal", "Not Solved") and prob.status != 1:
        return None
    squad_ids = [pid for pid, v in x.items() if (v.value() or 0) > 0.5]
    return pulp.value(prob.objective), squad_ids


def my_value(ids, gis):
    tot = 0.0
    for g in gis:
        t, _, _, b = best_xi(ids, g)
        tot += t + 0.1 * b
    return tot


def rating(my, best):
    # 100 = as good as the optimal squad for your budget; the floor (0) is set at 60% of optimal,
    # which is roughly a weak, poorly-structured team.
    if not best:
        return None
    return max(0, min(100, round((my / best - 0.6) / 0.4 * 100)))



def build_team(ME, league_ids, main):
    ent, picks, hist = entry_bundle(ME)
    squad = [p["element"] for p in picks["picks"]]
    eh = picks["entry_history"]
    bank = eh["bank"] / 10
    ft = free_transfers(hist)
    base_val = squad_value(squad)

    plan = []
    for gi, g in enumerate(gws):
        t, xi, cap, b = best_xi(squad, gi)
        ranked = sorted(xi, key=lambda i: -P[i]["xpts"][gi])
        plan.append({"gw": g, "xpts": round(t, 1), "xi": xi, "cap": cap, "vice": ranked[1] if len(ranked) > 1 else None})

    # transfer search -------------------------------------------------
    team_count = {}
    for i in squad:
        team_count[P[i]["team"]] = team_count.get(P[i]["team"], 0) + 1
    cands = {pos: sorted([p for p in players if p["pos"] == pos and p["id"] not in squad and p["avail"] > 0.5],
                         key=lambda p: -p["xh"])[:30] for pos in ("GKP", "DEF", "MID", "FWD")}


    def valid(new_ids, budget_left):
        if budget_left < -1e-9:
            return False
        tc = {}
        for i in new_ids:
            t = P[i]["team"]; tc[t] = tc.get(t, 0) + 1
            if tc[t] > 3:
                return False
        return True


    one = []
    for out in squad:
        po = P[out]
        for c in cands[po["pos"]]:
            new = [c["id"] if i == out else i for i in squad]
            left = bank + po["price"] - c["price"]
            if not valid(new, left):
                continue
            one.append((squad_value(new) - base_val, out, c["id"], left))
    one.sort(reverse=True)

    two = []
    top_outs = list(dict.fromkeys([o for _, o, _, _ in one[:60]] + sorted(squad, key=lambda i: P[i]["xh"])[:6]))
    for o1, o2 in combinations(top_outs, 2):
        p1, p2 = P[o1], P[o2]
        for c1 in cands[p1["pos"]][:18]:
            for c2 in cands[p2["pos"]][:18]:
                if c1["id"] == c2["id"]:
                    continue
                new = [c1["id"] if i == o1 else c2["id"] if i == o2 else i for i in squad]
                left = bank + p1["price"] + p2["price"] - c1["price"] - c2["price"]
                if not valid(new, left):
                    continue
                two.append((squad_value(new) - base_val, (o1, o2), (c1["id"], c2["id"]), left))
    two.sort(reverse=True)


    def fmt_moves(lst, n, hits):
        res, seen = [], set()
        for gain, outs, ins, left in lst:
            key = (tuple(sorted(outs if isinstance(outs, tuple) else (outs,))), tuple(sorted(ins if isinstance(ins, tuple) else (ins,))))
            if key in seen:
                continue
            seen.add(key)
            res.append({"out": list(outs) if isinstance(outs, tuple) else [outs],
                        "in": list(ins) if isinstance(ins, tuple) else [ins],
                        "gain": round(gain, 2), "hit": hits, "net": round(gain - hits, 2), "bank": round(left, 1)})
            if len(res) >= n:
                break
        return res


    hit2 = 0 if ft >= 2 else 4
    transfers = {"one": fmt_moves(one, 12, 0 if ft >= 1 else 4), "two": fmt_moves(two, 10, hit2)}

    # ------------------------------------------------------------------ mini-leagues
    leagues = []
    my_league_rank = {l["id"]: l for l in ent["leagues"]["classic"]}
    for lid in league_ids:
        st = get(f"leagues-classic/{lid}/standings/")
        rows = st["standings"]["results"][:50]

        def fetch(r):
            return r, *entry_bundle(r["entry"])[1:]

        data = pmap(fetch, rows, workers=6)
        members, own, capn = [], {}, {}
        for r, pk, hh in data:
            ids = [p["element"] for p in pk["picks"]] if pk else []
            cap = next((p["element"] for p in pk["picks"] if p["is_captain"]), None) if pk else None
            for i in ids:
                own[i] = own.get(i, 0) + 1
            if cap:
                capn[cap] = capn.get(cap, 0) + 1
            members.append({
                "entry": r["entry"], "team": r["entry_name"], "player": r["player_name"],
                "rank": r["rank"], "last_rank": r["last_rank"], "total": r["total"], "gw": r["event_total"],
                "squad": ids, "cap": cap, "chip": pk["active_chip"] if pk else None,
                "value": pk["entry_history"]["value"] / 10 if pk else None,
                "bank": pk["entry_history"]["bank"] / 10 if pk else None,
                "bench_pts": pk["entry_history"]["points_on_bench"] if pk else None,
                "hits": sum(x["event_transfers_cost"] for x in hh["current"]) if hh else None,
                "chips_left": [c["name"] for c in chips_left(hh)] if hh else [],
                "next_xpts": round(best_xi(ids, 0)[0], 1) if ids else None,
            })
        n = len(members)
        leagues.append({
            "id": lid, "name": st["league"]["name"], "size": n, "members": members,
            "own": {str(k): round(v / n * 100) for k, v in own.items()},
            "cap": {str(k): round(v / n * 100) for k, v in capn.items()},
        })

    budget_total = eh["value"] / 10 + bank
    ratings = None
    try:
        o1 = optimal_value([0], budget_total)
        o5 = optimal_value(list(range(min(5, len(gws)))), budget_total)
        if o1 and o5:
            m1, m5 = my_value(squad, [0]), my_value(squad, list(range(min(5, len(gws)))))
            ratings = {"gw": rating(m1, o1[0]), "outlook": rating(m5, o5[0]),
                       "gw_my": round(m1, 1), "gw_best": round(o1[0], 1),
                       "out_my": round(m5, 1), "out_best": round(o5[0], 1), "horizon": min(5, len(gws)),
                       "best_squad": o5[1]}
    except Exception as e:  # noqa
        print("ratings failed:", e, file=sys.stderr)

    # ------------------------------------------------------------------ rule-based advisor
    CHIP_BG = {"wildcard": "Wildcard", "freehit": "Free Hit", "bboost": "Bench Boost", "3xc": "Triple Captain"}
    THRESH = 2.5  # weighted xPts over horizon needed to justify spending a free transfer
    nm_ = lambda i: P[i]["name"]
    advice = []  # {lvl: good|warn|bad|info, cat, text}


    def add(lvl, cat, text):
        advice.append({"lvl": lvl, "cat": cat, "text": text})


    xi0 = set(plan[0]["xi"])
    g0 = gws[0]
    # red flags on my squad
    for i in squad:
        p = P[i]
        nfx = len(team_fx[p["team"]][g0])
        if nfx == 0:
            add("bad", "Отбор", f"{p['name']} няма мач в GW{g0}.")
        if (p["status"] != "a" or (p["cop"] is not None and p["cop"] < 100)) and p["status"] != "u":
            lvl = "bad" if (p["avail"] <= 0.5 and i in xi0) else "warn"
            add(lvl, "Отбор", f"{p['name']}: {p['news'] or 'под въпрос'}" + (" — в стартовия състав е." if i in xi0 else ""))
        if p["pc_pct"] <= -70:
            add("warn", "Цени", f"{p['name']} е близо до поевтиняване ({round(p['pc_pct'])}%). Ако ще го продаваш, по-добре преди нощната промяна.")

    # transfer verdict
    b1 = transfers["one"][0] if transfers["one"] else None
    b2 = transfers["two"][0] if transfers["two"] else None
    urgent = [i for i in xi0 if P[i]["avail"] <= 0.5]
    verdict = {"action": "roll", "moves": None}
    if b2 and ft >= 2 and b2["gain"] >= max(THRESH * 1.6, (b1["gain"] if b1 else 0) + 1.5):
        verdict = {"action": "two", "moves": b2}
    elif b1 and (b1["gain"] >= THRESH or (urgent and b1["gain"] >= 1.0) or (ft >= 5 and b1["gain"] > 0)):
        verdict = {"action": "one", "moves": b1}
    if verdict["action"] == "one" and b2 and ft < 2 and b2["net"] >= b1["gain"] + 2.0:
        verdict = {"action": "hit", "moves": b2}

    mv_txt = lambda m: ", ".join(f"{nm_(o)} → {nm_(n)}" for o, n in zip(m["out"], m["in"]))
    if verdict["action"] == "roll":
        best = f" Най-добрият вариант ({mv_txt(b1)}) носи само +{b1['gain']:.1f} за {len(gws)} кръга." if b1 else ""
        add("info", "Трансфер", f"Запази трансфера (ще имаш {min(ft + 1, 5)} следващия кръг).{best} Под прага от {THRESH} точки е.")
    elif verdict["action"] == "one":
        why = " (принудително заради контузия)" if urgent and b1["gain"] < THRESH else ""
        add("good", "Трансфер", f"Направи 1 трансфер{why}: {mv_txt(b1)} — около +{b1['gain']:.1f} очаквани точки за {len(gws)} кръга, банка след това £{b1['bank']:.1f}.")
    elif verdict["action"] == "two":
        add("good", "Трансфер", f"Използвай 2-та свободни трансфера: {mv_txt(b2)} — около +{b2['gain']:.1f} за {len(gws)} кръга.")
    else:
        add("good", "Трансфер", f"Хитът си струва: {mv_txt(b2)} — +{b2['gain']:.1f} брутно, +{b2['net']:.1f} нето след −4.")
    if verdict["action"] in ("one", "roll") and b2 and ft < 2 and verdict["action"] != "hit":
        if b2["net"] > 0:
            add("info", "Трансфер", f"Двоен ход с −4 ({mv_txt(b2)}) дава +{b2['net']:.1f} нето — малко над нулата, рискът от несигурност в модела не го оправдава.")
        else:
            add("info", "Трансфер", "Хит (−4) не се оправдава тази седмица.")
    if verdict["moves"]:
        for n in verdict["moves"]["in"]:
            if P[n]["pc_pct"] >= 70:
                add("warn", "Цени", f"{nm_(n)} е близо до поскъпване ({round(P[n]['pc_pct'])}%) — ако ще го взимаш, направи го преди нощната промяна.")

    # captain
    cap0 = plan[0]["cap"]
    my_rank = sorted(plan[0]["xi"], key=lambda i: -P[i]["xpts"][0])[:3]
    cap_line = ", ".join(f"{nm_(i)} {P[i]['xpts'][0]:.1f}" for i in my_rank)
    add("good", "Капитан", f"Капитан за GW{g0}: {nm_(cap0)}. Топ 3 по модела: {cap_line}.")
    for L in leagues:
        if L["cap"]:
            tc, pct = max(L["cap"].items(), key=lambda kv: kv[1])
            tc = int(tc)
            if tc != cap0 and pct >= 50 and tc in squad:
                diff = P[cap0]["xpts"][0] - P[tc]["xpts"][0]
                if diff < 0.6:
                    add("warn", "Капитан", f"В „{L['name']}“ {pct}% капитанстваха {nm_(tc)}. Разликата с {nm_(cap0)} е само {diff:.1f} точки — {nm_(tc)} е по-безопасният избор (щит), {nm_(cap0)} е залог за изпреварване.")

    # blanks / doubles / chips
    for g in gws:
        dbl = [TEAMS[t]["short_name"] for t in TEAMS if len(team_fx[t][g]) >= 2]
        blk = [TEAMS[t]["short_name"] for t in TEAMS if len(team_fx[t][g]) == 0]
        if dbl or blk:
            mine_blk = [nm_(i) for i in squad if len(team_fx[P[i]["team"]][g]) == 0]
            mine_dbl = [nm_(i) for i in squad if len(team_fx[P[i]["team"]][g]) >= 2]
            parts = []
            if dbl: parts.append(f"двоен кръг за {', '.join(dbl)}" + (f" (твои: {', '.join(mine_dbl)})" if mine_dbl else ""))
            if blk: parts.append(f"без мач: {', '.join(blk)}" + (f" (твои: {', '.join(mine_blk)})" if mine_blk else ""))
            add("info", "Чипове", f"GW{g}: " + "; ".join(parts) + ".")
    left = chips_left(hist)
    if left:
        until = left[0]["until"]
        names = ", ".join(CHIP_BG[c["name"]] for c in left)
        remaining = until - next_ev + 1
        add("warn" if remaining <= 4 else "info", "Чипове",
            f"Неизползвани чипове до GW{until}: {names} (остават {remaining} кръга). Изгарят, ако не ги ползваш до тогава.")

    # news feed (FPL official) for my squad + transfer targets + top owned
    watch = list(dict.fromkeys(squad + [i for m in transfers["one"][:5] for i in m["in"]]))
    news = []
    for e in sorted(els.values(), key=lambda e: e["news_added"] or "", reverse=True):
        if e["news"] and (e["id"] in watch or float(e["selected_by_percent"]) >= 10):
            news.append({"id": e["id"], "text": e["news"], "added": e["news_added"],
                         "mine": e["id"] in squad, "link": e.get("scout_news_link") or None})
    news = news[:40]

    # compact digest for the AI step (keep small!)
    digest = {
        "deadline": nxt["deadline_time"] if nxt else None, "next_gw": next_ev, "ft": ft, "bank": bank,
        "verdict": verdict["action"], "advice": [a["text"] for a in advice],
        "ratings": {k: v for k, v in (ratings or {}).items() if k != "best_squad"},
        "squad": [{"name": P[i]["name"], "team": TEAMS[P[i]["team"]]["short_name"], "pos": P[i]["pos"], "price": P[i]["price"],
                   "xp_next": P[i]["xpts"][0], "xp6": P[i]["xh"], "start": i in xi0, "news": P[i]["news"]} for i in squad],
        "targets": [{"name": nm_(i), "team": TEAMS[P[i]["team"]]["short_name"], "price": P[i]["price"], "xp6": P[i]["xh"]}
                    for i in dict.fromkeys(i for m in transfers["one"][:6] for i in m["in"])],
        "top_transfers": [mv_txt(m) + f" (+{m['gain']})" for m in transfers["one"][:5]],
        "leagues": [{"name": L["name"], "my_rank": next((m["rank"] for m in L["members"] if m["entry"] == ME), None),
                     "gap_to_1st": (L["members"][0]["total"] - next((m["total"] for m in L["members"] if m["entry"] == ME), 0)) if L["members"] else None}
                    for L in leagues],
    }
    if main:
        with open(os.path.join(ROOT, "docs", "digest.json"), "w", encoding="utf-8") as f:
            json.dump(digest, f, ensure_ascii=False, separators=(",", ":"))

    out = {
        "advice": advice, "verdict": verdict, "news": news, "ratings": ratings,
        "generated": datetime.now(timezone.utc).isoformat(timespec="minutes"),
        "current_gw": cur_ev, "next_gw": next_ev, "deadline": nxt["deadline_time"] if nxt else None,
        "gws": gws, "weights": WEIGHTS[: len(gws)], "league_avg_xg": round(MU, 2),
        "teams": {t: {"short": v["short_name"], "name": v["name"]} for t, v in TEAMS.items()},
        "players": players,
        "me": {
            "id": ME, "name": ent["name"], "player": f'{ent["player_first_name"]} {ent["player_last_name"]}',
            "overall_rank": ent["summary_overall_rank"], "points": ent["summary_overall_points"],
            "gw_points": ent["summary_event_points"],
            "squad": squad, "bank": bank, "value": eh["value"] / 10, "ft": ft,
            "chips_left": chips_left(hist), "plan": plan,
            "history": [{"gw": r["event"], "pts": r["points"], "rank": r["overall_rank"], "bench": r["points_on_bench"],
                         "hit": r["event_transfers_cost"]} for r in hist["current"]],
            "transfers": transfers,
            "league_ranks": {str(l): {"rank": my_league_rank.get(l, {}).get("entry_rank")} for l in league_ids},
        },
        "leagues": leagues,
        "ticker": ticker,
    }

    fname = "data.json" if main else f"data-{ME}.json"
    with open(os.path.join(ROOT, "docs", fname), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    print("ok", ME, len(players), "players;", "gws", gws, "ft", ft, "bank", bank)


TEAMS_CFG = CFG.get("teams") or [{"id": CFG["team_id"], "leagues": CFG.get("leagues", [])}]
for k, tcfg in enumerate(TEAMS_CFG):
    try:
        build_team(tcfg["id"], tcfg.get("leagues", []), main=(k == 0))
    except Exception as e:  # noqa
        print("team failed", tcfg["id"], e, file=sys.stderr)
        if k == 0:
            raise
