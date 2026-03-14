import os
import json
import csv
import re
import numpy as np
from scipy.stats import spearmanr
import networkx as nx
from collections import defaultdict
from itertools import combinations

# ============================================================
# Swift static analysis (baseline type-level metric analysis)
# ============================================================

TYPE_DEF_PATTERN = re.compile(
    r'\b(class|struct|enum|protocol|actor)\s+(\w+)(?:\s*:\s*([^{]+))?\s*\{',
    re.MULTILINE
)

# Captures: func name, parameter list, optional return type
FUNC_SIG_PATTERN = re.compile(
    r'\bfunc\s+(\w+)\s*\(([^)]*)\)\s*(?:->\s*([^{\s]+))?',
    re.MULTILINE
)

# Stored property declarations (best-effort static approximation)
PROP_DECL_PATTERN = re.compile(
    r'\b(var|let)\s+(\w+)\s*:\s*([\w<>\[\]\.]+\??)',
    re.MULTILINE
)

# Property type annotations like: var x: TypeName
PROPERTY_TYPE_PATTERN = re.compile(
    r'\b(var|let)\s+\w+\s*:\s*([\w<>\[\]\.]+\??)',
    re.MULTILINE
)

# Qualified static call like: TypeName.method(
QUALIFIED_CALL_PATTERN = re.compile(r'\b(\w+)\.(\w+)\s*\(', re.MULTILINE)

# Any call-like token: foo(
CALL_PATTERN = re.compile(r'\b(\w+)\s*\(', re.MULTILINE)

# Control-flow decision points for CC (best-effort)
DECISION_TOKEN_PATTERN = re.compile(
    r'\bif\b|\belse\s+if\b|\bfor\b|\bwhile\b|\bcase\b|\bcatch\b|\bguard\b|\brepeat\b|\bswitch\b|\b&&\b|\b\|\|\b|\?\s*:',
    re.MULTILINE
)

# Remove comments/strings to reduce false positives
LINE_COMMENT = re.compile(r'//.*?$',
                          re.MULTILINE)
BLOCK_COMMENT = re.compile(r'/\*.*?\*/',
                           re.DOTALL)
STRING_LIT = re.compile(r'"(?:\\.|[^"\\])*"', re.DOTALL)

def _strip_swift_noise(s: str) -> str:
    s = BLOCK_COMMENT.sub('', s)
    s = LINE_COMMENT.sub('', s)
    s = STRING_LIT.sub('""', s)
    return s

def _brace_extract_block(src: str, start_idx: int) -> tuple[str, int] | tuple[None, None]:
    """
    Extract {...} block starting at or after start_idx.
    Returns (block_text_including_braces, end_idx_inclusive) or (None, None).
    """
    i = start_idx
    while i < len(src) and src[i] != '{':
        i += 1
    if i >= len(src):
        return None, None

    brace = 0
    j = i
    while j < len(src):
        if src[j] == '{':
            brace += 1
        elif src[j] == '}':
            brace -= 1
            if brace == 0:
                return src[i:j+1], j
        j += 1
    return None, None

def find_swift_files(root_dir: str) -> list[str]:
    swift_files = []
    for base, _, files in os.walk(root_dir):
        for fn in files:
            if fn.endswith(".swift"):
                swift_files.append(os.path.join(base, fn))
    return swift_files

def extract_types_and_bodies(swift_files: list[str]) -> dict[str, str]:
    """
    Merge multiple definitions/extensions by type name into a single concatenated body text.
    This is not full SwiftSyntax merging, but helps approximate extension contributions.
    """
    types = defaultdict(list)
    for file in swift_files:
        try:
            with open(file, encoding="utf-8", errors="ignore") as f:
                content = _strip_swift_noise(f.read())
        except Exception:
            continue

        for match in TYPE_DEF_PATTERN.finditer(content):
            type_name = match.group(2)
            block, end_idx = _brace_extract_block(content, match.end() - 1)
            if block:
                # Store the full declaration block (header + body)
                start = match.start()
                decl = content[start:end_idx+1]
                types[type_name].append(decl)

        # Extensions: `extension TypeName { ... }` (best effort)
        for ext_match in re.finditer(r'\bextension\s+(\w+)[^{]*\{', content, re.MULTILINE):
            type_name = ext_match.group(1)
            block, end_idx = _brace_extract_block(content, ext_match.end() - 1)
            if block:
                start = ext_match.start()
                decl = content[start:end_idx+1]
                types[type_name].append(decl)

    return {t: "\n\n".join(chunks) for t, chunks in types.items()}

def build_type_dependency_graph(types: dict[str, str]) -> nx.DiGraph:
    """
    Directed edge T -> U if T has a statically identifiable structural dependency on U.
    """
    G = nx.DiGraph()
    for t in types:
        G.add_node(t)

    type_names = set(types.keys())

    for t, body in types.items():
        # Inheritance / protocol conformance clause from first matching type header
        m = TYPE_DEF_PATTERN.search(body)
        if m and m.group(3):
            parents = [p.strip() for p in m.group(3).split(",")]
            for p in parents:
                if p in type_names:
                    G.add_edge(t, p)

        # Property types
        for _, prop_type in PROPERTY_TYPE_PATTERN.findall(body):
            base = prop_type.replace("?", "")
            base = base.split("<", 1)[0].split("[", 1)[0].split(".", 1)[0]
            if base in type_names:
                G.add_edge(t, base)

        # Function signatures: parameter types + return types
        for func_name, params, ret in FUNC_SIG_PATTERN.findall(body):
            if ret:
                base = ret.replace("?", "")
                base = base.split("<", 1)[0].split("[", 1)[0].split(".", 1)[0]
                if base in type_names:
                    G.add_edge(t, base)

            for param in params.split(","):
                if ":" in param:
                    ptype = param.split(":", 1)[1].strip()
                    base = ptype.replace("?", "")
                    base = base.split("<", 1)[0].split("[", 1)[0].split(".", 1)[0]
                    if base in type_names:
                        G.add_edge(t, base)

        # Qualified static calls / type usage: TypeName.method(
        for owner, _method in QUALIFIED_CALL_PATTERN.findall(body):
            if owner in type_names:
                G.add_edge(t, owner)

        # Explicit instantiation: TypeName(
        for tok in CALL_PATTERN.findall(body):
            if tok in type_names:
                G.add_edge(t, tok)

    return G

def extract_functions_by_type(types: dict[str, str]) -> dict[str, list[dict]]:
    """
    Returns mapping type -> list of function dicts:
    { "name": str, "body": str, "cc": int }
    """
    out = defaultdict(list)

    for t, body in types.items():
        for m in FUNC_SIG_PATTERN.finditer(body):
            fname = m.group(1)
            # Find the function body block starting at the first '{' after signature
            block, end_idx = _brace_extract_block(body, m.end())
            if not block:
                continue
            # strip outer braces for scanning
            inner = block[1:-1]

            # CC_f = 1 + (# decision tokens)
            decisions = len(DECISION_TOKEN_PATTERN.findall(inner))
            cc = 1 + decisions

            out[t].append({"name": fname, "body": inner, "cc": cc})

    return out

def build_complete_call_graph(types: dict[str, str],
                              funcs_by_type: dict[str, list[dict]]) -> nx.DiGraph:
    """
    Refined static call graph approximation.

    Node: (TypeName, funcName)
    Edge: (T,f) -> (U,g) if statically identifiable invocation exists.

    Handles:
        - self.method()
        - TypeName.method()
        - varName.method() (best-effort resolution)
        - unqualified method() calls
    """

    call_g = nx.DiGraph()

    # --------------------------------------------------------
    # Index function ownership
    # --------------------------------------------------------
    func_owners = defaultdict(set)   # func_name -> {Type1, Type2}
    for t, funcs in funcs_by_type.items():
        for f in funcs:
            node = (t, f["name"])
            call_g.add_node(node)
            func_owners[f["name"]].add(t)

    type_names = set(types.keys())

    # --------------------------------------------------------
    # Build variable-to-type map (very conservative)
    # --------------------------------------------------------
    var_type_map = defaultdict(dict)  # type -> {varName: TypeName}

    VAR_DECL_PATTERN = re.compile(
        r'\b(var|let)\s+(\w+)\s*:\s*([\w<>\[\]\.]+\??)'
    )

    for t, body in types.items():
        for _, var_name, var_type in VAR_DECL_PATTERN.findall(body):
            base = var_type.replace("?", "")
            base = base.split("<", 1)[0].split("[", 1)[0].split(".", 1)[0]
            if base in type_names:
                var_type_map[t][var_name] = base

    # --------------------------------------------------------
    # Scan function bodies
    # --------------------------------------------------------
    QUALIFIED_PATTERN = re.compile(r'\b(\w+)\.(\w+)\s*\(')
    SIMPLE_CALL_PATTERN = re.compile(r'\b(\w+)\s*\(')

    for t, funcs in funcs_by_type.items():
        for f in funcs:
            src = (t, f["name"])
            body = f["body"]

            # ---- Qualified calls: X.method(
            for owner_token, method_name in QUALIFIED_PATTERN.findall(body):

                # Case 1: TypeName.method()
                if owner_token in type_names:
                    if method_name in func_owners:
                        for real_owner in func_owners[method_name]:
                            call_g.add_edge(src, (real_owner, method_name))

                # Case 2: self.method()
                elif owner_token == "self":
                    if method_name in func_owners and t in func_owners[method_name]:
                        call_g.add_edge(src, (t, method_name))

                # Case 3: varName.method()
                elif owner_token in var_type_map[t]:
                    inferred_type = var_type_map[t][owner_token]
                    if method_name in func_owners and inferred_type in func_owners[method_name]:
                        call_g.add_edge(src, (inferred_type, method_name))

            # ---- Unqualified calls: method(
            for token in SIMPLE_CALL_PATTERN.findall(body):

                if token in func_owners:

                    # Prefer same-type resolution first
                    if t in func_owners[token]:
                        call_g.add_edge(src, (t, token))
                    else:
                        # Otherwise connect to all possible owners
                        for owner in func_owners[token]:
                            call_g.add_edge(src, (owner, token))

    return call_g

def compute_LCOT(type_name: str,
                 funcs: list[dict],
                 stored_props: set[str]) -> int:
    """
    Best-effort LCOM-like: pairs of functions not sharing properties minus pairs sharing.
    Property usage is approximated by presence of 'prop' or 'self.prop' tokens.
    """
    if len(funcs) < 2 or not stored_props:
        return 0

    used = []
    for f in funcs:
        body = f["body"]
        used_props = set()
        for p in stored_props:
            if re.search(rf'\bself\.{re.escape(p)}\b', body) or re.search(rf'\b{re.escape(p)}\b', body):
                used_props.add(p)
        used.append(used_props)

    P = 0
    S = 0
    for i, j in combinations(range(len(funcs)), 2):
        if used[i] & used[j]:
            S += 1
        else:
            P += 1

    return P - S

def compute_RFT(type_name: str,
                funcs_by_type: dict[str, list[dict]],
                call_g: nx.DiGraph) -> int:
    owned = {(type_name, f["name"]) for f in funcs_by_type.get(type_name, [])}
    owned = {n for n in owned if call_g.has_node(n)}

    invoked = set()
    for n in owned:
        invoked.update(call_g.successors(n))   # direct calls only

    return len(owned | invoked)

def third_quartile(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.array(values, dtype=float), 75))

def summarize_graph(G: nx.DiGraph, name: str, snapshot_name: str) -> dict:
    summary = {}

    n = G.number_of_nodes()
    m = G.number_of_edges()

    summary["snapshot"] = snapshot_name
    summary["graph"] = name
    summary["nodes"] = n
    summary["edges"] = m

    summary["density"] = nx.density(G) if n > 1 else 0.0

    if n > 0:
        degrees = [deg for _, deg in G.degree()]
        summary["avg_degree"] = float(np.mean(degrees))
        summary["max_degree"] = int(max(degrees))
    else:
        summary["avg_degree"] = 0.0
        summary["max_degree"] = 0

    # Weakly connected components
    wcc = list(nx.weakly_connected_components(G))
    summary["wcc_count"] = len(wcc)
    summary["largest_wcc_size"] = max((len(c) for c in wcc), default=0)

    # Strongly connected components
    scc = list(nx.strongly_connected_components(G))
    summary["scc_count"] = len(scc)
    summary["largest_scc_size"] = max((len(c) for c in scc), default=0)

    # Diameter & average shortest path (largest WCC only)
    if wcc:
        largest_comp = max(wcc, key=len)
        sub = G.subgraph(largest_comp).copy()
        UG = sub.to_undirected()

        if len(UG) > 1 and nx.is_connected(UG):
            try:
                summary["diameter"] = nx.diameter(UG)
                summary["avg_shortest_path"] = nx.average_shortest_path_length(UG)
            except:
                summary["diameter"] = -1
                summary["avg_shortest_path"] = -1
        else:
            summary["diameter"] = 0
            summary["avg_shortest_path"] = 0
    else:
        summary["diameter"] = 0
        summary["avg_shortest_path"] = 0

    return summary

def analyze_snapshot(snapshot_dir: str,
                     weights: dict[str, float] | None = None,
                     callgraph_dir: str = None):

    graph_summaries = []

    snapshot_name = os.path.basename(snapshot_dir)

    swift_files = find_swift_files(snapshot_dir)
    types = extract_types_and_bodies(swift_files)

    if not types:
        return [], []

    # ----------------------------------------------------
    # Dependency graph (used only for CBT metric)
    # ----------------------------------------------------

    G = build_type_dependency_graph(types)

    graph_summaries.append(
        summarize_graph(G, "FullGraph", snapshot_name)
    )

    funcs_by_type = extract_functions_by_type(types)
    call_g = build_complete_call_graph(types, funcs_by_type)

    if callgraph_dir:
        nx.write_gexf(
            call_g,
            os.path.join(callgraph_dir, f"{snapshot_name}_callgraph.gexf")
        )

    type_names = list(types.keys())

    # ----------------------------------------------------
    # Stored properties
    # ----------------------------------------------------

    props_by_type = {}

    for t, body in types.items():
        props = set()
        for _kind, name, _ptype in PROP_DECL_PATTERN.findall(body):
            props.add(name)
        props_by_type[t] = props

    # ----------------------------------------------------
    # Compute baseline metrics
    # ----------------------------------------------------

    CC_T = {}
    WMT_T = {}
    LCOT_T = {}
    CBT_T = {}
    RFT_T = {}

    # CC_T and WMT_T
    for t in type_names:

        funcs = funcs_by_type.get(t, [])

        if funcs:
            cc_vals = [f["cc"] for f in funcs]
            CC_T[t] = float(np.mean(cc_vals))
            WMT_T[t] = float(np.sum(cc_vals))
        else:
            CC_T[t] = 0.0
            WMT_T[t] = 0.0

        LCOT_T[t] = float(
            compute_LCOT(t, funcs, props_by_type.get(t, set()))
        )

    # CBT_T
    for t in type_names:
        CBT_T[t] = float(len(set(G.successors(t))))

    # RFT_T
    for t in type_names:
        RFT_T[t] = float(
            compute_RFT(t, funcs_by_type, call_g)
        )

    # ----------------------------------------------------
    # Thresholds
    # ----------------------------------------------------

    D_CC = [CC_T[t] for t in type_names]
    D_WMT = [WMT_T[t] for t in type_names]
    D_LCOT = [LCOT_T[t] for t in type_names]
    D_CBT = [CBT_T[t] for t in type_names]
    D_RFT = [RFT_T[t] for t in type_names]

    tau = {}

    tau["CC_T"] = max(10.0, third_quartile(D_CC))
    tau["WMT_T"] = max(20.0, third_quartile(D_WMT))
    tau["LCOT_T"] = third_quartile(D_LCOT)
    tau["CBT_T"] = third_quartile(D_CBT)
    tau["RFT_T"] = third_quartile(D_RFT)

    # ----------------------------------------------------
    # Metric weights
    # ----------------------------------------------------

    metric_keys = ["CC_T", "WMT_T", "LCOT_T", "CBT_T", "RFT_T"]

    if weights is None:
        w = {k: 1.0 / len(metric_keys) for k in metric_keys}
    else:
        w = {k: float(weights.get(k, 0.0)) for k in metric_keys}
        s = sum(w.values())
        w = {k: v / s for k, v in w.items()} if s > 0 else {k: 1.0/len(metric_keys) for k in metric_keys}

    # ----------------------------------------------------
    # Excess computation
    # ----------------------------------------------------

    def excess_val(metric_name, t):

        val = {
            "CC_T": CC_T[t],
            "WMT_T": WMT_T[t],
            "LCOT_T": LCOT_T[t],
            "CBT_T": CBT_T[t],
            "RFT_T": RFT_T[t],
        }[metric_name]

        return max(0.0, val - tau[metric_name])

    total_excess = {}

    for t in type_names:

        total_excess[t] = sum(
            w[m] * excess_val(m, t)
            for m in metric_keys
        )

    # ----------------------------------------------------
    # Baseline prioritization
    # ----------------------------------------------------

    ranking = []

    for t in type_names:

        ranking.append({

            "type": t,
            "severity": total_excess[t],

            "CC_T": CC_T[t],
            "WMT_T": WMT_T[t],
            "LCOT_T": LCOT_T[t],
            "CBT_T": CBT_T[t],
            "RFT_T": RFT_T[t],

            "tau_CC_T": tau["CC_T"],
            "tau_WMT_T": tau["WMT_T"],
            "tau_LCOT_T": tau["LCOT_T"],
            "tau_CBT_T": tau["CBT_T"],
            "tau_RFT_T": tau["RFT_T"]
        })

    ranking.sort(
        key=lambda x: (-x["severity"], x["type"])
    )

    return ranking, graph_summaries
# ============================================================
# Longitudinal evaluation metrics (sensitivity + consistency)
# ============================================================

def find_snapshots(parent_dir, project_name, snapshot_prefix):

    project_root = os.path.join(parent_dir, project_name)

    if not os.path.isdir(project_root):
        print(f"ERROR: '{project_name}' directory not found under {parent_dir}")
        return []

    return sorted([
        d for d in os.listdir(project_root)
        if d.startswith(snapshot_prefix)
        and os.path.isdir(os.path.join(project_root, d))
    ])

def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)

def run_project(project_name, snapshot_prefix, output_root):

    parent = os.getcwd()

    OUTPUT_ROOT = output_root
    CALLGRAPH_DIR = None

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    snapshots = find_snapshots(parent, project_name, snapshot_prefix)

    reader_output = {}
    sensitivity_rows = []
    consistency_rows = []
    all_graph_summaries = []

    previous = None

    for snapshot in snapshots:

        snapshot_path = os.path.join(parent, project_name, snapshot)

        print(f"[{project_name}] Analyzing {snapshot}...")

        results, graph_summaries = analyze_snapshot(
            snapshot_path,
            callgraph_dir=CALLGRAPH_DIR
        )

        reader_output[snapshot] = results

        types = [r["type"] for r in results]
        rank_scores = [r["severity"] for r in results]
        debt_types = {r["type"] for r in results if r["severity"] > 0}

        all_graph_summaries.extend(graph_summaries)

        total_mag = sum(rank_scores)
        distribution = [s / total_mag for s in rank_scores] if total_mag > 0 else []

        if previous:

            v_now = np.array(distribution, dtype=float)
            v_prev = np.array(previous["distribution"], dtype=float)

            max_len = max(len(v_now), len(v_prev))
            v_now = np.pad(v_now, (0, max_len - len(v_now)))
            v_prev = np.pad(v_prev, (0, max_len - len(v_prev)))

            distribution_shift = float(np.sum(np.abs(v_now - v_prev)))

            top_now = set(types[:10])
            top_prev = set(previous["types"][:10])
            turnover = 1 if top_now != top_prev else 0

            membership_shift = float(1 - jaccard(debt_types, previous["debt_types"]))

            sensitivity_value = (distribution_shift + turnover + membership_shift) / 3.0

            sensitivity_rows.append({
                "snapshot": snapshot,
                "distribution_shift": distribution_shift,
                "turnover": turnover,
                "membership_shift": membership_shift,
                "sensitivity": sensitivity_value
            })

            min_len = min(len(rank_scores), len(previous["rank_scores"]))

            if min_len > 1:
                rho, _ = spearmanr(
                    rank_scores[:min_len],
                    previous["rank_scores"][:min_len]
                )
                rho = float(rho) if not np.isnan(rho) else 1.0
            else:
                rho = 1.0

            top_now = set(types[:10])
            top_prev = set(previous["types"][:10])
            overlap = len(top_now & top_prev) / len(top_now | top_prev)

            var_now = float(np.var(rank_scores)) if rank_scores else 0.0
            var_prev = previous["variance"]
            variance_shift = abs(var_now - var_prev)

            variance_stability = 1.0 / (1.0 + variance_shift)

            consistency_value = (rho + overlap + variance_stability) / 3.0

            consistency_rows.append({
                "snapshot": snapshot,
                "rank_correlation": rho,
                "topk_overlap": overlap,
                "variance_shift": variance_shift,
                "variance_stability": variance_stability,
                "consistency": consistency_value
            })

        previous = {
            "types": types,
            "debt_types": debt_types,
            "distribution": distribution,
            "rank_scores": rank_scores,
            "variance": float(np.var(rank_scores)) if rank_scores else 0.0
        }

    # ==========================================================
    # Write Outputs
    # ==========================================================

    # 1) JSON (reader-facing type ranking per snapshot)
    with open(os.path.join(OUTPUT_ROOT, "type_results.json"), "w", encoding="utf-8") as f:
        json.dump(reader_output, f, indent=2)

    # 2) Type Ranking CSV
    ranking_rows = []

    for snapshot in sorted(reader_output.keys()):
        ranked_types = reader_output[snapshot]

        for rank, t in enumerate(ranked_types, start=1):

            ranking_rows.append({
                "snapshot": snapshot,
                "rank": rank,
                "type": t["type"],
                "severity": t["severity"],
                "CC_T": t["CC_T"],
                "WMT_T": t["WMT_T"],
                "LCOT_T": t["LCOT_T"],
                "CBT_T": t["CBT_T"],
                "RFT_T": t["RFT_T"]
            })

    with open(os.path.join(OUTPUT_ROOT, "type_ranking.csv"),
            "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "snapshot",
                "rank",
                "type",
                "severity",
                "CC_T",
                "WMT_T",
                "LCOT_T",
                "CBT_T",
                "RFT_T"
                ]
        )
        writer.writeheader()
        writer.writerows(ranking_rows)
        
    # 3) Sensitivity CSV
    with open(os.path.join(OUTPUT_ROOT, "sensitivity.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "snapshot",
            "distribution_shift",
            "turnover",
            "membership_shift",
            "sensitivity"
        ])
        writer.writeheader()
        writer.writerows(sensitivity_rows)

    # 4) Consistency CSV
    with open(os.path.join(OUTPUT_ROOT, "consistency.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "snapshot",
                "rank_correlation",
                "topk_overlap",
                "variance_shift",
                "variance_stability",
                "consistency"
            ]
        )
        writer.writeheader()
        writer.writerows(consistency_rows)

    # 5) Summary CSV
    summary = {
        "mean_distribution_shift": float(np.mean(
            [r["distribution_shift"] for r in sensitivity_rows]
        )) if sensitivity_rows else 0.0,

        "mean_turnover": float(np.mean(
            [r["turnover"] for r in sensitivity_rows]
        )) if sensitivity_rows else 0.0,

        "mean_membership_shift": float(np.mean(
            [r["membership_shift"] for r in sensitivity_rows]
        )) if sensitivity_rows else 0.0,

        "mean_rank_correlation": float(np.mean(
            [r["rank_correlation"] for r in consistency_rows]
        )) if consistency_rows else 0.0,

        "mean_topk_overlap": float(np.mean(
            [r["topk_overlap"] for r in consistency_rows]
        )) if consistency_rows else 0.0,

        "mean_variance_shift": float(np.mean(
            [r["variance_shift"] for r in consistency_rows]
        )) if consistency_rows else 0.0,

        # Paper composite metrics
        "composite_sensitivity": float(np.mean(
            [r["sensitivity"] for r in sensitivity_rows]
        )) if sensitivity_rows else 0.0,

        "composite_consistency": float(np.mean(
            [r["consistency"] for r in consistency_rows]
        )) if consistency_rows else 0.0
    }

    with open(os.path.join(OUTPUT_ROOT, "summary_metrics.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in summary.items():
            writer.writerow([k, v])

    print("\nGenerated files in:", OUTPUT_ROOT)
    print(" - type_results.json")
    print(" - type_ranking.csv")
    print(" - sensitivity.csv")
    print(" - consistency.csv")
    print(" - summary_metrics.csv")
    print(" - graph_structure_summary.csv")

    # 6) Graph Structure Summary CSV
    with open(os.path.join(OUTPUT_ROOT, "graph_structure_summary.csv"),
            "w", newline="", encoding="utf-8") as f:

        fieldnames = [
            "snapshot",
            "graph",
            "nodes",
            "edges",
            "density",
            "avg_degree",
            "max_degree",
            "wcc_count",
            "largest_wcc_size",
            "scc_count",
            "largest_scc_size",
            "diameter",
            "avg_shortest_path"
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_graph_summaries)
# ============================================================
# Main driver: produces JSON + CSVs
# ============================================================

def main():

    # ----------------------------------------
    # Baseline output folder
    # ----------------------------------------
    base_output = "baseline type analysis"

    run_project(
        project_name="MenuCal",
        snapshot_prefix="MenuCal_backup_",
        output_root=os.path.join(base_output, "menucal")
    )

    run_project(
        project_name="NutriCompass",
        snapshot_prefix="NutriCompass_backup_",
        output_root=os.path.join(base_output, "nutricompass")
    )

if __name__ == "__main__":
    main()