"""chipagent.py - ChipAgent: agentic research copilot (End-to-End System).
Roles (design section 9.1): Planner -> Executor -> Critic -> Librarian -> Gatekeeper.
 * Planner   : turns a request into a schema-checked plan (LoRA-fine-tuned open LLM when available, rule-based otherwise); plan shown before execution.
 * Executor  : whitelisted deterministic tool wrappers with retries and provenance; no free-form code execution.
 * Critic    : rule checks (ranges, QC flags, uncertainty, split-gap, citation support).
 * Librarian : hybrid dense + keyword retrieval over the downloaded PMC / LiverTox XML with licence tags and a quote-length cap.
 * Gatekeeper: any physical/irreversible action needs an approval token (cfg.approvals) which is logged.
Also: LLM fine-tuning (LoRA) on instruction data derived from the pipeline's own outputs, explanations, and next-experiment proposals."""
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

import chem
from utils import LOG, Timer, tq, force_cleanup, save_json, finish_plot

ASSAY_MECHANISM = {   # public Tox21 assay definitions
    "SR-MMP": "disruption of mitochondrial membrane potential (mitochondrial toxicity)",
    "SR-ARE": "activation of the Nrf2/ARE antioxidant response (oxidative / electrophilic stress)",
    "SR-p53": "p53 pathway activation (DNA damage / genotoxic stress)",
    "SR-HSE": "heat-shock factor response (proteotoxic stress)",
    "SR-ATAD5": "ATAD5 stabilisation (genotoxicity / replication stress)",
    "NR-AhR": "aryl hydrocarbon receptor activation",
    "NR-Aromatase": "aromatase inhibition (endocrine disruption)",
    "NR-ER": "estrogen receptor activation", "NR-AR": "androgen receptor activation",
    "NR-PPAR-gamma": "PPAR-gamma activation", "composite": "general cytotoxicity / loss of viability",
}
IRREVERSIBLE = {"order_compound", "start_physical_experiment", "modify_hardware", "dispense_liquid"}


# =========================================================================== Librarian
class Librarian:
    def __init__(self, cfg, res):
        self.cfg = cfg
        self.docs = []
        self._load(res)
        self._index()

    def _load(self, res):
        from lxml import etree
        files = [(p, "pmc") for p in res.pmc_xml()] + [(p, "livertox") for p in res.livertox_nxml()[:600]]
        for p, kind in tq(files, "Librarian: parsing XML"):
            try:
                t = etree.parse(p, etree.XMLParser(recover=True, huge_tree=True)).getroot()
            except Exception:
                continue
            txt = lambda e: " ".join("".join(e.itertext()).split())
            title = next((txt(x) for x in t.iter("article-title")), os.path.basename(p))
            pmcid = next((txt(x) for x in t.iter("article-id") if x.get("pub-id-type") in ("pmcid", "pmc")), os.path.splitext(os.path.basename(p))[0])
            lic = next((x.get("{http://www.w3.org/1999/xlink}href") or txt(x) for x in t.iter("license")), "unknown")
            lic_tag = "CC-BY (reuse ok)" if re.search(r"/by/|publicdomain|/zero/", str(lic)) and "nc" not in str(lic).lower() else ("non-commercial / restricted" if lic != "unknown" else "unknown")
            paras = [txt(x) for x in t.iter("p")]
            buf = ""
            for para in paras:
                buf = (buf + " " + para).strip()
                if len(buf.split()) >= 110:
                    self.docs.append(dict(id=str(pmcid), title=title, text=buf, license=lic_tag, kind=kind)); buf = ""
            if len(buf.split()) >= 40:
                self.docs.append(dict(id=str(pmcid), title=title, text=buf, license=lic_tag, kind=kind))
        LOG.info(f"[librarian] {len(self.docs):,} passages from {len(files)} documents")

    def _index(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1, max_features=200000, sublinear_tf=True)
        self.M = self.tfidf.fit_transform([d["title"] + " " + d["text"] for d in self.docs]) if self.docs else None
        self.emb, self.st = None, None
        try:
            from sentence_transformers import SentenceTransformer
            self.st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda" if _cuda() else "cpu")
            self.emb = self.st.encode([d["text"][:1500] for d in self.docs], batch_size=64, normalize_embeddings=True, show_progress_bar=True)
        except Exception as e:
            LOG.warning(f"[librarian] dense encoder unavailable ({e!r}); using LSA as the dense channel")
            from sklearn.decomposition import TruncatedSVD
            if self.M is not None and self.M.shape[0] > 5:
                k = min(128, self.M.shape[0] - 1, self.M.shape[1] - 1)
                self.svd = TruncatedSVD(k, random_state=0).fit(self.M)
                E = self.svd.transform(self.M); self.emb = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)

    def search(self, query, k=5):
        if not self.docs:
            return []
        q = self.tfidf.transform([query])
        kw = (self.M @ q.T).toarray().ravel()
        if self.st is not None:
            qe = self.st.encode([query], normalize_embeddings=True)[0]
        else:
            qe = self.svd.transform(q)[0]; qe = qe / (np.linalg.norm(qe) + 1e-9)
        de = self.emb @ qe if self.emb is not None else np.zeros(len(self.docs))
        rk = lambda s: np.argsort(np.argsort(-s))
        rrf = 1 / (60 + rk(kw)) + 1 / (60 + rk(de))                       # reciprocal-rank fusion (hybrid)
        out, seen = [], set()
        for i in np.argsort(-rrf):
            d = self.docs[i]
            if d["id"] in seen:
                continue
            seen.add(d["id"])
            out.append(dict(id=d["id"], title=d["title"][:140], license=d["license"], score=float(rrf[i]), snippet=self._snip(d["text"], query)))
            if len(out) >= k:
                break
        return out

    def _snip(self, text, query):
        """Quote-limited evidence: the best-matching sentence, truncated to cfg.quote_word_limit words."""
        sents = re.split(r"(?<=[.!?])\s+", text)
        qs = set(re.findall(r"\w{4,}", query.lower()))
        best = max(sents, key=lambda s: len(qs & set(re.findall(r"\w{4,}", s.lower()))))
        w = best.split()
        return " ".join(w[: self.cfg.quote_word_limit]) + (" ..." if len(w) > self.cfg.quote_word_limit else "")


def _cuda():
    try:
        import torch; return torch.cuda.is_available()
    except Exception:
        return False


# =========================================================================== Planner / Executor / Critic / Gatekeeper
@dataclass
class Step:
    id: int
    tool: str
    inputs: dict
    expected: str
    depends_on: list = field(default_factory=list)


@dataclass
class Plan:
    goal: str
    steps: list
    def to_json(self): return json.dumps(asdict(self), indent=2)


TOOLS = ("plate_qc", "dose_response", "score_toxicity", "translate_brightfield", "segment_cells", "phenotype", "retrieve_literature", "critic_review", "propose_experiment")


def validate_plan(obj):
    """Schema check used for both LLM-generated and rule-based plans."""
    try:
        if not isinstance(obj, dict) or "steps" not in obj or not obj["steps"]:
            return False
        ids = set()
        for s in obj["steps"]:
            if s["tool"] not in TOOLS or "id" not in s or "expected" not in s:
                return False
            if any(d not in ids for d in s.get("depends_on", [])):
                return False
            ids.add(s["id"])
        return True
    except Exception:
        return False


def rule_plan(request):
    r = request.lower()
    steps = []
    add = lambda tool, inp, exp, dep=None: steps.append(Step(len(steps) + 1, tool, inp, exp, dep or []))
    add("plate_qc", {}, "per-plate Z' / edge / drift flags")
    if any(k in r for k in ("dose", "ec50", "curve", "concentration")):
        add("dose_response", {}, "4PL fits with CI and quality flags", [1])
    if any(k in r for k in ("tox", "safe", "liver", "kidney", "compound", "drug", "smiles")):
        add("score_toxicity", {"smiles": re.findall(r"smiles\s*[:=]\s*([^\s,;]+)", request, flags=re.I)}, "toxicity probabilities with uncertainty", [1])
    if any(k in r for k in ("bright", "fluor", "label-free", "translate", "image")):
        add("translate_brightfield", {}, "virtual fluorescence + trust map")
    if any(k in r for k in ("segment", "cell", "phenotype", "morpholog", "image")):
        add("segment_cells", {}, "instance masks"); add("phenotype", {}, "cell-state clusters / MoA")
    add("retrieve_literature", {"query": request}, "cited mechanistic evidence")
    add("critic_review", {}, "QC/range/citation checks", [s.id for s in steps])
    add("propose_experiment", {}, "next experiment (human approval required)", [len(steps)])
    return Plan(request, steps)


class Gatekeeper:
    def __init__(self, cfg): self.cfg, self.log = cfg, []
    def request(self, action, detail):
        tok = self.cfg.approvals.get(action)
        ok = bool(tok) and action in IRREVERSIBLE
        self.log.append(dict(t=time.strftime("%F %T"), action=action, detail=detail, approved=ok, token=(tok[:4] + "***") if tok else None))
        return ok


class Executor:
    """Whitelisted tool calls. Each call: retry (2x), provenance record (inputs hash, duration, status)."""
    def __init__(self, tools): self.tools, self.prov, self.results = tools, [], {}
    def run(self, plan, verbose=True):
        for st in tq(plan.steps, "ChipAgent: executing plan"):
            if st.tool not in self.tools:
                self.prov.append(dict(step=st.id, tool=st.tool, status="REJECTED (not whitelisted)")); continue
            t0, err = time.time(), None
            for attempt in range(3):
                try:
                    self.results[st.id] = self.tools[st.tool](**st.inputs); err = None; break
                except Exception as e:
                    err = repr(e); time.sleep(0.5)
            self.prov.append(dict(step=st.id, tool=st.tool, inputs_sha=hashlib.sha1(json.dumps(st.inputs, sort_keys=True, default=str).encode()).hexdigest()[:10],
                                  seconds=round(time.time() - t0, 2), status="ok" if err is None else f"failed: {err[:120]}"))
        return self.results


class Critic:
    def __init__(self, cfg, unc_thresh=None): self.cfg, self.unc_thresh = cfg, unc_thresh
    def review(self, tox_rows, plate_qc, evidence, gaps):
        issues = []
        if tox_rows is not None and len(tox_rows):
            if not tox_rows.tox_score.between(0, 1).all(): issues.append(("ERROR", "toxicity score outside [0,1]"))
            thr = self.unc_thresh or float(tox_rows.tox_uncertainty.quantile(0.75))
            n = int((tox_rows.tox_uncertainty > thr).sum())
            if n: issues.append(("WARN", f"{n} compound(s) have high predictive uncertainty (> {thr:.3f}); treat as 'inconclusive'"))
        if plate_qc is not None and len(plate_qc):
            bad = plate_qc[~plate_qc.passed]
            if len(bad): issues.append(("WARN", f"{len(bad)}/{len(plate_qc)} plates failed QC (e.g. {bad.plate.iloc[0]}: {bad.qc_flags.iloc[0]}); results from them are down-weighted"))
        for name, g in gaps.items():
            if g and g.get("overfit"): issues.append(("WARN", f"{name}: train-holdout gap {g['gap_train_holdout']:.3f} > {self.cfg.overfit_gap_warn}"))
        if not evidence: issues.append(("WARN", "no literature evidence retrieved: mechanistic statements are model-derived only"))
        return issues


# =========================================================================== explanation + proposals
def explain_compound(out_tox, idx_holdout, topk=4):
    """Gradient x input attribution of the composite toxicity head -> top contributing fingerprint bits / descriptors -> substructures."""
    import torch
    import models as M
    net = out_tox["deep"]["cnn_attn"].to("cpu").eval()
    X = torch.as_tensor(out_tox["Xc"]["holdout"][idx_holdout:idx_holdout + 1], dtype=torch.float32).requires_grad_(True)
    tasks = out_tox["tasks"]
    cols = [tasks.index(a) for a in ("tox21:SR-MMP", "tox21:SR-ARE", "tox21:SR-p53", "composite:cytotox") if a in tasks] or list(range(min(4, len(tasks))))
    net.zero_grad(); net(X)[0, cols].sum().backward()
    att = (X.grad * X).detach().numpy()[0]
    names = out_tox["prep"].names_
    smi = out_tox["df"].iloc[out_tox["splits"]["holdout"][idx_holdout]].std_smiles
    res = []
    for j in np.argsort(-np.abs(att))[:topk * 3]:
        n = names[j]
        frag = None
        if n.startswith("fp"):
            frag = chem.substructure_for_bit(smi, int(n[2:]))
            if not frag: continue
        res.append(dict(feature=n, attribution=float(att[j]), substructure=frag))
        if len(res) >= topk: break
    return res


def propose_next_experiment(tox_rows, dose_fits, cfg, n=5):
    """Uncertainty- and boundary-driven active learning with scaffold diversity; dose range from fitted EC50s when available."""
    if tox_rows is None or not len(tox_rows):
        return []
    t = tox_rows.copy()
    t["boundary"] = 1 - np.abs(t.tox_score - 0.5) * 2
    t["acq"] = (t.tox_uncertainty / (t.tox_uncertainty.max() + 1e-9)) * 0.6 + t.boundary * 0.4
    pick, scaf = [], set()
    for _, r in t.sort_values("acq", ascending=False).iterrows():
        if r.scaffold in scaf: continue
        scaf.add(r.scaffold); pick.append(r)
        if len(pick) >= n: break
    lo, hi = -2.0, 1.0
    if dose_fits is not None and len(dose_fits) and "logEC50" in dose_fits:
        e = dose_fits[dose_fits.fit_ok == True].logEC50.dropna()
        if len(e): lo, hi = float(e.quantile(0.1)) - 0.7, float(e.quantile(0.9)) + 0.7
    doses = np.round(10 ** np.linspace(lo, hi, 8), 4).tolist()
    return [dict(smiles=r.std_smiles, scaffold=r.scaffold, tox_score=float(r.tox_score), uncertainty=float(r.tox_uncertainty), acquisition=float(r.acq),
                 suggested_doses_uM=doses, replicates=3, controls="DMSO (neg) + a plate-matched positive control; randomise well positions to avoid edge effects",
                 rationale="high model uncertainty near the decision boundary - labelling it is the most informative next measurement",
                 status="PROPOSAL ONLY - requires human approval via Gatekeeper before any physical action") for r in pick]


# =========================================================================== LLM fine-tuning (LoRA)
def _hf_token(cfg):
    try:
        t = open(cfg.hf_token_path).read().strip()
        if t:
            return t
    except Exception:
        pass
    from utils import get_secret
    return get_secret("HF_TOKEN") or get_secret("HUGGINGFACE_TOKEN")


def build_instruction_data(out_tox, dose, cfg, librarian=None):
    """Instruction pairs derived from the pipeline's real outputs. Held-out prompts = the holdout compound split."""
    rng = np.random.default_rng(cfg.seed)
    hold = out_tox["holdout_scores"]
    tasks = out_tox["tasks"]
    train_rows, eval_rows = [], []
    reqs = ["Assess the toxicity of this compound and explain it.", "Is this compound likely to be hepatotoxic in a liver chip?", "Summarise the mechanistic risk for this molecule.", "What should we worry about with this compound in a chip assay?"]
    def narrative(r):
        heads = {a.split(":")[-1]: r.get(f"p:{a}") for a in tasks if f"p:{a}" in r.index and a.startswith("tox21:SR")}
        top = sorted(heads.items(), key=lambda kv: -(kv[1] if kv[1] == kv[1] else 0))[:2]
        mech = "; ".join(f"{k} ({ASSAY_MECHANISM.get(k, k)}) p={v:.2f}" for k, v in top)
        lvl = "HIGH" if r.tox_score > 0.7 else "MODERATE" if r.tox_score > 0.4 else "LOW"
        unc = "inconclusive (high model uncertainty)" if r.tox_uncertainty > hold.tox_uncertainty.quantile(0.75) else "reasonably confident"
        return f"Composite toxicity score {r.tox_score:.2f} -> {lvl} risk, {unc}. Leading mechanistic signals: {mech}. Recommend confirmatory dose-response with a viability counter-screen."
    for i, r in hold.iterrows():
        ex = dict(instruction=str(rng.choice(reqs)), input=f"SMILES: {r.std_smiles}", output=narrative(r), key=r.ik14)
        (eval_rows if rng.random() < 0.3 else train_rows).append(ex)
    plan_prompts = ["Assess toxicity of these compounds on a liver chip", "Run QC then fit dose-response curves for my plate", "Translate bright-field images to fluorescence and segment the cells", "Profile phenotype and mechanism of action of my treated chips"]
    for k in range(60):
        q = str(rng.choice(plan_prompts)); p = rule_plan(q)
        (eval_rows if k % 4 == 0 else train_rows).append(dict(instruction="Return a JSON analysis plan using only the allowed tools.", input=q, output=p.to_json(), key=f"plan{k}", is_plan=True))
    return train_rows, eval_rows


def train_lora(model, tok, tr, ev, cfg, steps, out_dir):
    """LoRA fine-tune `model` on instruction rows; returns metrics. Dynamic padding, loss only on the response tokens."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments
    tok.pad_token = tok.pad_token or tok.eos_token
    prompt = lambda e: f"### Instruction:\n{e['instruction']}\n\n### Input:\n{e['input']}\n\n### Response:\n"
    class DS(torch.utils.data.Dataset):
        def __init__(s, rows): s.r = rows
        def __len__(s): return len(s.r)
        def __getitem__(s, i):
            e = s.r[i]
            p = tok(prompt(e), add_special_tokens=False)["input_ids"]
            r = tok(e["output"] + (tok.eos_token or ""), add_special_tokens=False)["input_ids"]
            ids = (p + r)[:640]
            lab = ([-100] * len(p) + r)[:640]
            return dict(input_ids=ids, labels=lab)
    def collate(batch):
        L = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), L), tok.pad_token_id, dtype=torch.long); lab = torch.full((len(batch), L), -100, dtype=torch.long); att = torch.zeros((len(batch), L), dtype=torch.long)
        for k, b in enumerate(batch):
            n = len(b["input_ids"]); ids[k, :n] = torch.tensor(b["input_ids"]); lab[k, :n] = torch.tensor(b["labels"]); att[k, :n] = 1
        return dict(input_ids=ids, attention_mask=att, labels=lab)
    targets = sorted({n.split(".")[-1] for n, _ in model.named_modules() if n.split(".")[-1] in ("q_proj", "k_proj", "v_proj", "o_proj", "c_attn")})
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.1, target_modules=targets, task_type="CAUSAL_LM"))
    for p in model.parameters():                       # fp16 base + trainable fp32 adapters (avoids 'unscale FP16 gradients' errors)
        if p.requires_grad:
            p.data = p.data.float()
    kw = dict(output_dir=str(out_dir), per_device_train_batch_size=2, per_device_eval_batch_size=2, gradient_accumulation_steps=4, learning_rate=2e-4, max_steps=steps,
              warmup_steps=max(1, steps // 20), weight_decay=0.01, logging_steps=10, save_strategy="no", fp16=torch.cuda.is_available(), report_to=[],
              lr_scheduler_type="cosine", max_grad_norm=1.0, remove_unused_columns=False, dataloader_pin_memory=False)
    try:
        args = TrainingArguments(eval_strategy="steps", eval_steps=max(10, steps // 5), **kw)
    except TypeError:
        args = TrainingArguments(evaluation_strategy="steps", eval_steps=max(10, steps // 5), **kw)
    trn = Trainer(model=model, args=args, train_dataset=DS(tr), eval_dataset=DS(ev[:80]), data_collator=collate)
    trn.train()
    ev_loss = float(trn.evaluate()["eval_loss"])
    model.save_pretrained(out_dir); tok.save_pretrained(out_dir)
    ok = tot = 0
    model.eval()
    for e in [x for x in ev if x.get("is_plan")][:12]:
        enc = tok(prompt(e), return_tensors="pt").to(model.device)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=500, do_sample=False, pad_token_id=tok.pad_token_id)
        txt = tok.decode(g[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        try:
            ok += int(validate_plan(json.loads(txt[: txt.rfind("}") + 1])))
        except Exception:
            pass
        tot += 1
    return dict(eval_loss=ev_loss, perplexity=float(np.exp(ev_loss)), plan_json_valid=f"{ok}/{tot}", train_examples=len(tr), eval_examples=len(ev))


def finetune_llm(out_tox, dose, cfg):
    """LoRA fine-tune of the first open, permissively licensed (Apache-2.0 / MIT, not gated) model that is reachable.
    Returns a status dict instead of raising when the Hub / GPU / libraries are unavailable."""
    if not cfg.run_llm_finetune:
        return dict(status="disabled in config")
    try:
        import torch
        from huggingface_hub import model_info, login
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import peft  # noqa: F401
    except Exception as e:
        return dict(status=f"skipped: LLM libraries unavailable ({e!r})")
    tok_ = _hf_token(cfg)
    if tok_:
        try:
            login(token=tok_, add_to_git_credential=False)
        except Exception:
            tok_ = None
    chosen = None
    for name in cfg.llm_candidates:
        try:
            mi = model_info(name, token=tok_)
            lic = (getattr(mi.card_data, "license", None) or "").lower() if mi.card_data else ""
            if lic in cfg.allowed_licenses and not getattr(mi, "gated", False):
                chosen = (name, lic); break
            LOG.warning(f"[llm] {name} rejected (licence={lic!r}, gated={getattr(mi, 'gated', None)})")
        except Exception as e:
            LOG.warning(f"[llm] cannot inspect {name}: {e!r}")
    if not chosen:
        return dict(status="skipped: no candidate model passed the open-licence check (or Hub unreachable)")
    name, lic = chosen
    LOG.info(f"[llm] selected {name} (licence {lic})")
    steps = cfg.llm_max_steps if torch.cuda.is_available() else min(cfg.llm_max_steps, 30)
    tr, ev = build_instruction_data(out_tox, dose, cfg)
    tok = AutoTokenizer.from_pretrained(name, token=tok_)
    dt = torch.float16 if torch.cuda.is_available() else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(name, token=tok_, dtype=dt)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(name, token=tok_, torch_dtype=dt)
    if torch.cuda.is_available():
        model = model.to("cuda")
    try:
        m = train_lora(model, tok, tr, ev, cfg, steps, cfg.models / "llm_lora")
    finally:
        del model
        force_cleanup()
    return dict(status="ok", model=name, license=lic, steps=steps, **m)


# =========================================================================== orchestration
class ChipAgent:
    def __init__(self, cfg, res, ctx):
        self.cfg, self.res, self.ctx = cfg, res, ctx
        self.lib = Librarian(cfg, res)
        self.gate = Gatekeeper(cfg)
        tox = ctx.get("tox")
        self.tools = {
            "plate_qc": lambda: ctx["dose"]["plate_qc"],
            "dose_response": lambda: ctx["dose"].get("fits"),
            "score_toxicity": lambda smiles=None: self._score(smiles),
            "translate_brightfield": lambda: ctx["glow"]["metrics"] if ctx.get("glow") else None,
            "segment_cells": lambda: ctx["cell"]["seg_summary"] if ctx.get("cell") else None,
            "phenotype": lambda: (ctx["cell"].get("moa") if ctx.get("cell") else None),
            "retrieve_literature": lambda query: self.lib.search(query, 5),
            "critic_review": lambda: None, "propose_experiment": lambda: None,
        }

    def _score(self, smiles):
        tox = self.ctx.get("tox")
        if smiles:
            from chiptox import ToxScorer
            return ToxScorer(self.cfg).score([s for s in smiles if s])
        return tox["holdout_scores"] if tox else None

    def run(self, request, n_compounds=8, llm_plan=None):
        t0 = time.time()
        plan = rule_plan(request)
        if llm_plan and validate_plan(llm_plan): plan = Plan(request, [Step(**s) for s in llm_plan["steps"]])
        LOG.info("[agent] PLAN (shown before execution):\n" + plan.to_json())
        ex = Executor(self.tools); results = ex.run(plan)
        tox = self.ctx.get("tox"); hold = tox["holdout_scores"] if tox else None
        evid = {}
        rows = []
        if hold is not None:
            sel = pd.concat([hold.sort_values("tox_score", ascending=False).head(max(1, n_compounds // 2)),
                             hold.sort_values("tox_uncertainty", ascending=False).head(max(1, n_compounds // 2))]).drop_duplicates("ik14")
            for i, r in sel.iterrows():
                heads = {a.split(":")[-1]: r[f"p:{a}"] for a in tox["tasks"] if f"p:{a}" in r.index and a.startswith("tox21:SR")}
                top = sorted(heads.items(), key=lambda kv: -kv[1])[:2]
                expl = explain_compound(tox, int(hold.index.get_loc(i)))
                q = " ".join(ASSAY_MECHANISM.get(k, k) for k, _ in top) + " organ-on-a-chip toxicity"
                ev = self.lib.search(q, 3); evid[r.ik14] = ev
                rows.append(dict(smiles=r.std_smiles, tox_score=r.tox_score, uncertainty=r.tox_uncertainty, top_mechanisms=[(k, ASSAY_MECHANISM.get(k, k), float(v)) for k, v in top],
                                 substructure_drivers=[e for e in expl if e.get("substructure")][:3], evidence=ev, true_cytotox=r.get("y:composite:cytotox", np.nan)))
        gaps = {"ChipTox": (tox or {}).get("gap", {}).get((tox or {}).get("best_name")), **{f"GlowFromLight/{k}": v for k, v in ((self.ctx.get("glow") or {}).get("gap") or {}).items()}}
        issues = Critic(self.cfg).review(hold.iloc[:0].assign() if False else (pd.DataFrame(rows).rename(columns={"uncertainty": "tox_uncertainty"})[["tox_score", "tox_uncertainty"]] if rows else None),
                                         self.ctx["dose"]["plate_qc"] if self.ctx.get("dose") else None, evid, gaps)
        prop = propose_next_experiment(hold, (self.ctx.get("dose") or {}).get("fits"), self.cfg)
        approvals = [dict(proposal=p["smiles"], approved=self.gate.request("start_physical_experiment", p["smiles"])) for p in prop[:1]]
        report = self._report(request, plan, rows, issues, prop, ex.prov, approvals)
        path = self.cfg.reports / "chipscreen_agent_report.md"; path.write_text(report)
        save_json(dict(plan=asdict(plan), provenance=ex.prov, gatekeeper=self.gate.log, issues=issues, proposals=prop), self.cfg.reports / "agent_run_provenance.json")
        LOG.info(f"[agent] done in {time.time()-t0:.1f}s -> {path}")
        return dict(report=report, path=str(path), plan=plan, compounds=rows, issues=issues, proposals=prop, provenance=ex.prov)

    def _report(self, request, plan, rows, issues, prop, prov, approvals):
        c = self.ctx; L = [f"# ChipScreen report\n\n**Request:** {request}\n", "## Plan (shown before execution)\n"]
        L += [f"{s.id}. `{s.tool}` -> {s.expected}" for s in plan.steps]
        L.append("\n## Toxicity (unseen holdout compounds)\n")
        for r in rows:
            L.append(f"- `{r['smiles']}`  score **{r['tox_score']:.2f}**, uncertainty {r['uncertainty']:.3f}")
            L.append("  - mechanisms: " + "; ".join(f"{k} - {m} (p={p:.2f})" for k, m, p in r["top_mechanisms"]))
            if r["substructure_drivers"]: L.append("  - model-attributed substructures: " + ", ".join(f"`{d['substructure']}`" for d in r["substructure_drivers"]))
            for e in r["evidence"]: L.append(f"  - evidence [{e['id']}] ({e['license']}): \"{e['snippet']}\"")
        for k, name in (("tox", "ChipTox"), ("glow", "GlowFromLight"), ("cell", "ChipCell"), ("dose", "DoseCurve")):
            if c.get(k) is None: L.append(f"\n## {name}\nnot run")
        if c.get("dose") and c["dose"].get("plate_qc") is not None: L.append(f"\n## Plate QC\n{len(c['dose']['plate_qc'])} plates; median Z' = {c['dose']['plate_qc'].zprime.median():.2f}")
        L.append("\n## Critic\n" + ("\n".join(f"- **{lv}**: {m}" for lv, m in issues) or "- no issues"))
        L.append("\n## Proposed next experiments (human approval required)\n")
        for p in prop: L.append(f"- `{p['smiles']}` score {p['tox_score']:.2f}, unc {p['uncertainty']:.3f}; doses uM {p['suggested_doses_uM']}; x{p['replicates']} replicates. {p['rationale']}")
        L.append("\n## Gatekeeper\n" + "\n".join(f"- {a['proposal']}: {'APPROVED' if a['approved'] else 'NOT approved (no approval token) - nothing executed'}" for a in approvals))
        L.append("\n## Provenance\n" + "\n".join(f"- step {p['step']} {p['tool']} {p['status']} ({p.get('seconds','-')}s)" for p in prov))
        return "\n".join(L)
