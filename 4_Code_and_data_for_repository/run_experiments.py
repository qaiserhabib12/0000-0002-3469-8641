"""
run_experiments.py -- Unified-protocol experiments for the revised paddy-disease paper.

Every number, table and results figure in the revised manuscript is produced by this
script. Nothing in the paper should be typed by hand.

HOW TO RUN (Google Colab, GPU runtime, e.g. T4):
    !pip -q install kaggle
    # upload kaggle.json, accept the competition rules on kaggle.com, then:
    !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
    !kaggle competitions download -c paddy-disease-classification -p /content
    !unzip -q /content/paddy-disease-classification.zip -d /content/paddy

    !python run_experiments.py --stage train      # all configs x all seeds (resumable)
    !python run_experiments.py --stage embed      # embeddings for representation analysis
    !python run_experiments.py --stage submit     # Kaggle submission.csv from the selected model
    # submit submission.csv on Kaggle (late submission), read the PRIVATE score, then:
    !python run_experiments.py --stage report --kaggle_private 0.XXXX --kaggle_public 0.XXXX

Then copy everything in OUT_DIR (results.tex, tab_*.tex, fig_*.png) next to main.tex
and recompile. All [TBD] markers in the PDF are replaced automatically.

Training is resumable: each (config, seed) run writes runs/<config>_s<seed>.json and is
skipped if that file already exists, so a dropped Colab session loses at most one run.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- config
CFG = dict(
    data_dir="/content/paddy",
    out_dir="/content/paper_results",
    img_size=224,              # one resolution for every model
    batch=32,
    split_seed=2026,           # fixed stratified split, shared by all models
    val_frac=0.15,
    test_frac=0.15,
    seeds=[0, 1, 2, 3, 4],     # 5 seeds; reduce to [0, 1, 2] if GPU time is short
    head_epochs=10,            # stage 1 (backbone frozen)
    ft_epochs=30,              # stage 2 (partial unfreeze); frozen configs get the same total budget (40)
    head_lr=1e-3,
    ft_lr=1e-4,
    cnn_epochs=60,             # custom CNN: upper bound, early stopping decides
    cnn_lr=1e-3,
    es_patience=6,
    plateau_patience=3,
    plateau_factor=0.5,
    min_lr=1e-7,
    mnv2_unfreeze_from=100,    # MobileNetV2: layers [100:] trainable in stage 2
    resnet_unfreeze_block="conv5_",  # ResNet50: conv5 stage trainable in stage 2
    head_dropout_1=0.2,
    head_units=256,
    head_dropout_2=0.3,
    cnn_dropout=0.5,
    cluster_seeds=[0, 1, 2, 3, 4],
    cluster_pca_dims=50,
    cluster_n_neighbors=15,
    time_budget_h=10.5,        # stop starting new runs after this, so later stages finish within Kaggle's 12 h
)

CONFIGS = ["cnn", "mnv2_frozen", "mnv2_ft", "r50_frozen", "r50_ft"]
PRETTY = {
    "cnn": "Custom CNN",
    "mnv2_frozen": "MobileNetV2 (frozen)",
    "mnv2_ft": "MobileNetV2 (fine-tuned)",
    "r50_frozen": "ResNet50 (frozen)",
    "r50_ft": "ResNet50 (fine-tuned)",
}
CLASS_PRETTY = {
    "bacterial_leaf_blight": "Bacterial leaf blight",
    "bacterial_leaf_streak": "Bacterial leaf streak",
    "bacterial_panicle_blight": "Bacterial panicle blight",
    "blast": "Blast",
    "brown_spot": "Brown spot",
    "dead_heart": "Dead heart",
    "downy_mildew": "Downy mildew",
    "hispa": "Hispa",
    "normal": "Normal",
    "tungro": "Tungro",
}


def out(*p):
    path = os.path.join(CFG["out_dir"], *p)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


# --------------------------------------------------------------------------- split
def get_split():
    """Fixed stratified 70/15/15 split, written once to splits.csv and reused by all models."""
    from sklearn.model_selection import train_test_split

    path = out("splits.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    df = pd.read_csv(os.path.join(CFG["data_dir"], "train.csv"))
    df["path"] = [os.path.join(CFG["data_dir"], "train_images", l, i)
                  for l, i in zip(df.label, df.image_id)]
    classes = sorted(df.label.unique())
    df["y"] = df.label.map({c: k for k, c in enumerate(classes)})
    tr, tmp = train_test_split(df, test_size=CFG["val_frac"] + CFG["test_frac"],
                               stratify=df.y, random_state=CFG["split_seed"])
    rel = CFG["test_frac"] / (CFG["val_frac"] + CFG["test_frac"])
    va, te = train_test_split(tmp, test_size=rel, stratify=tmp.y,
                              random_state=CFG["split_seed"])
    tr, va, te = tr.assign(split="train"), va.assign(split="val"), te.assign(split="test")
    df = pd.concat([tr, va, te]).reset_index(drop=True)
    df.to_csv(path, index=False)
    return df


# --------------------------------------------------------------------------- tf parts
def tf_setup():
    import tensorflow as tf
    return tf


def preprocess_fn(config):
    tf = tf_setup()
    if config == "cnn":
        return lambda x: x / 255.0
    if config.startswith("mnv2"):
        return tf.keras.applications.mobilenet_v2.preprocess_input   # -> [-1, 1]
    return tf.keras.applications.resnet50.preprocess_input           # caffe mean subtraction


def make_ds(df, config, training, seed):
    """Augmentation is applied ONLY to the training split. Val/test are deterministic."""
    tf = tf_setup()
    S, B = CFG["img_size"], CFG["batch"]
    AUT = tf.data.AUTOTUNE
    ds = tf.data.Dataset.from_tensor_slices((df.path.values, df.y.values.astype("int32")))

    def load(p, y):
        img = tf.io.decode_jpeg(tf.io.read_file(p), channels=3)
        img = tf.image.resize(img, (S, S))
        return tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8), y

    ds = ds.map(load, num_parallel_calls=AUT).cache()   # uint8 cache keeps RAM low
    if training:
        ds = ds.shuffle(len(df), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(B)
    prep = preprocess_fn(config)
    if training:
        aug = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal", seed=seed),
            tf.keras.layers.RandomRotation(20 / 360, fill_mode="reflect", seed=seed),
            tf.keras.layers.RandomTranslation(0.1, 0.1, fill_mode="reflect", seed=seed),
            tf.keras.layers.RandomZoom(0.15, fill_mode="reflect", seed=seed),
            tf.keras.layers.RandomContrast(0.1, seed=seed),
        ])
        ds = ds.map(lambda x, y: (tf.clip_by_value(aug(tf.cast(x, tf.float32), training=True), 0, 255), y),
                    num_parallel_calls=AUT)
    else:
        ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=AUT)
    ds = ds.map(lambda x, y: (prep(x), y), num_parallel_calls=AUT)
    return ds.prefetch(AUT)


def build_model(config, n_classes, weights="imagenet"):
    tf = tf_setup()
    L = tf.keras.layers
    S = CFG["img_size"]
    inp = L.Input((S, S, 3))
    if config == "cnn":
        x = inp
        for f in (32, 64, 128):
            x = L.Conv2D(f, 3, padding="same", use_bias=False)(x)
            x = L.BatchNormalization()(x)
            x = L.ReLU()(x)
            x = L.MaxPooling2D()(x)
        x = L.GlobalAveragePooling2D(name="gap")(x)
        x = L.Dense(128, activation="relu", name="embedding")(x)
        x = L.Dropout(CFG["cnn_dropout"])(x)
        outp = L.Dense(n_classes, activation="softmax")(x)
        return tf.keras.Model(inp, outp), None
    if config.startswith("mnv2"):
        base = tf.keras.applications.MobileNetV2(include_top=False, weights=weights, input_shape=(S, S, 3))
    else:
        base = tf.keras.applications.ResNet50(include_top=False, weights=weights, input_shape=(S, S, 3))
    base.trainable = False
    x = base(inp, training=False)        # BatchNorm stays in inference mode in both stages
    x = L.GlobalAveragePooling2D(name="gap")(x)
    x = L.Dropout(CFG["head_dropout_1"])(x)
    x = L.Dense(CFG["head_units"], activation="relu", name="embedding")(x)
    x = L.Dropout(CFG["head_dropout_2"])(x)
    outp = L.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model(inp, outp), base


def unfreeze(base, config):
    tf = tf_setup()
    base.trainable = True
    if config.startswith("mnv2"):
        start = CFG["mnv2_unfreeze_from"]
    else:
        start = next(i for i, l in enumerate(base.layers) if l.name.startswith(CFG["resnet_unfreeze_block"]))
    for i, l in enumerate(base.layers):
        l.trainable = i >= start and not isinstance(l, tf.keras.layers.BatchNormalization)
    return start


def callbacks():
    tf = tf_setup()
    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=CFG["es_patience"],
                                         restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=CFG["plateau_factor"],
                                             patience=CFG["plateau_patience"], min_lr=CFG["min_lr"]),
    ]


def count_params(model):
    import tensorflow as tf
    tr = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    nt = int(sum(np.prod(w.shape) for w in model.non_trainable_weights))
    return tr, nt


def train_one(config, seed, df, classes):
    tf = tf_setup()
    res_path = out("runs", f"{config}_s{seed}.json")
    if os.path.exists(res_path):
        print("skip", res_path)
        return
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    ds_tr = make_ds(tr, config, True, seed)
    ds_va = make_ds(va, config, False, seed)
    ds_te = make_ds(te, config, False, seed)
    model, base = build_model(config, len(classes))
    loss = "sparse_categorical_crossentropy"
    hist, phase_boundary, params = {}, None, {}

    def fit(lr, epochs):
        model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=loss, metrics=["accuracy"])
        h = model.fit(ds_tr, validation_data=ds_va, epochs=epochs, callbacks=callbacks(), verbose=2)
        for k, v in h.history.items():
            hist.setdefault(k, []).extend([float(a) for a in v])

    if config == "cnn":
        params["stage1"] = count_params(model)
        fit(CFG["cnn_lr"], CFG["cnn_epochs"])
    elif config.endswith("_frozen"):
        params["stage1"] = count_params(model)
        fit(CFG["head_lr"], CFG["head_epochs"] + CFG["ft_epochs"])       # matched total budget
    else:
        params["stage1"] = count_params(model)
        fit(CFG["head_lr"], CFG["head_epochs"])
        phase_boundary = len(hist["loss"])
        params["unfreeze_from_layer"] = unfreeze(base, config)
        params["stage2"] = count_params(model)
        fit(CFG["ft_lr"], CFG["ft_epochs"])

    # evaluation with deterministic, unshuffled pipelines -> labels stay aligned
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                 precision_recall_fscore_support)
    res = dict(config=config, seed=seed, history=hist, phase_boundary=phase_boundary,
               params=params, classes=classes)
    for name, part, ds in (("val", va, ds_va), ("test", te, ds_te)):
        prob = model.predict(ds, verbose=0)
        pred = prob.argmax(1)
        y = part.y.values
        p, r, f, s = precision_recall_fscore_support(y, pred, labels=range(len(classes)), zero_division=0)
        nll = float(-np.mean(np.log(np.clip(prob[np.arange(len(y)), y], 1e-12, 1))))
        res[name] = dict(acc=float(accuracy_score(y, pred)),
                         macro_f1=float(f1_score(y, pred, average="macro")),
                         weighted_f1=float(f1_score(y, pred, average="weighted")),
                         loss=nll, precision=p.tolist(), recall=r.tolist(), f1=f.tolist(),
                         support=s.tolist(),
                         cm=confusion_matrix(y, pred, labels=range(len(classes))).tolist(),
                         pred=pred.tolist())
    json.dump(res, open(res_path, "w"))
    if seed == CFG["seeds"][0]:
        model.save(out("models", f"{config}_s{seed}.keras"))
    print(config, seed, "test acc %.4f macroF1 %.4f" % (res["test"]["acc"], res["test"]["macro_f1"]))


def stage_train():
    import time
    t0 = time.time()
    df = get_split()
    classes = sorted(df.label.unique())
    for seed in CFG["seeds"]:
        for c in CONFIGS:
            hrs = (time.time() - t0) / 3600
            if hrs > CFG["time_budget_h"]:
                print(f"TIME BUDGET REACHED ({hrs:.2f} h): not starting {c} seed {seed}")
                return
            train_one(c, seed, df, classes)
    print(f"ALL RUNS FINISHED in {(time.time() - t0) / 3600:.2f} h")


# --------------------------------------------------------------------------- selection
def load_runs():
    runs = {}
    for f in sorted(glob.glob(out("runs", "*.json"))):
        r = json.load(open(f))
        runs.setdefault(r["config"], []).append(r)
    return runs


def select_best(runs):
    """Model selection uses VALIDATION macro-F1 only. The test split is never used to choose."""
    scores = {c: np.mean([r["val"]["macro_f1"] for r in rs]) for c, rs in runs.items()}
    order = sorted(scores, key=scores.get, reverse=True)
    return order[0], order[1]


def stage_embed():
    """Embeddings of the full TEST split for representation analysis, incl. baselines."""
    tf = tf_setup()
    df = get_split()
    te = df[df.split == "test"]
    np.save(out("emb", "labels.npy"), te.y.values)
    s0 = CFG["seeds"][0]

    def extract(model, layer, config, tag):
        f = tf.keras.Model(model.input, model.get_layer(layer).output)
        X = f.predict(make_ds(te, config, False, 0), verbose=0)
        np.save(out("emb", f"{tag}.npy"), X)
        print(tag, X.shape)

    tf.keras.utils.set_random_seed(0)
    rnd, _ = build_model("r50_frozen", 10, weights=None)
    extract(rnd, "gap", "r50_frozen", "r50_random_gap")
    img, _ = build_model("r50_frozen", 10, weights="imagenet")
    extract(img, "gap", "r50_frozen", "r50_imagenet_gap")
    ft = tf.keras.models.load_model(out("models", f"r50_ft_s{s0}.keras"))
    extract(ft, "gap", "r50_ft", "r50_ft_gap")
    extract(ft, "embedding", "r50_ft", "r50_ft_head")
    cnn = tf.keras.models.load_model(out("models", f"cnn_s{s0}.keras"))
    extract(cnn, "embedding", "cnn", "cnn_head")


def stage_submit():
    tf = tf_setup()
    runs = load_runs()
    best, _ = select_best(runs)
    model = tf.keras.models.load_model(out("models", f"{best}_s{CFG['seeds'][0]}.keras"))
    classes = runs[best][0]["classes"]
    files = sorted(glob.glob(os.path.join(CFG["data_dir"], "test_images", "*.jpg")))
    d = pd.DataFrame({"path": files, "y": 0})
    prob = model.predict(make_ds(d, best, False, 0), verbose=0)
    sub = pd.DataFrame({"image_id": [os.path.basename(f) for f in files],
                        "label": [classes[k] for k in prob.argmax(1)]})
    sub.to_csv(out("submission.csv"), index=False)
    print("wrote", out("submission.csv"), "using", best)


# --------------------------------------------------------------------------- report
def pm(vals, pct=True, nd=2):
    v = np.array(vals, float) * (100 if pct else 1)
    return f"${v.mean():.{nd}f} \\pm {v.std(ddof=1) if len(v) > 1 else 0:.{nd}f}$"


def mean_s(vals, pct=True, nd=2):
    v = np.array(vals, float) * (100 if pct else 1)
    return f"{v.mean():.{nd}f}"


def best_epoch(h):
    return int(np.argmin(h["val_loss"]))


def mcnemar(a_pred, b_pred, y):
    from scipy.stats import binomtest
    a, b = np.array(a_pred) == y, np.array(b_pred) == y
    n01, n10 = int(np.sum(a & ~b)), int(np.sum(~a & b))
    if n01 + n10 == 0:
        return 1.0
    return float(binomtest(n01, n01 + n10, 0.5).pvalue)   # exact McNemar


def run_clustering():
    from sklearn.cluster import KMeans, SpectralClustering
    from sklearn.decomposition import PCA
    from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                                 normalized_mutual_info_score, silhouette_score)
    from sklearn.preprocessing import StandardScaler

    y = np.load(out("emb", "labels.npy"))
    sources = [("r50_random_gap", "ResNet50, random init (GAP)"),
               ("r50_imagenet_gap", "ResNet50, ImageNet only (GAP)"),
               ("r50_ft_gap", "ResNet50, fine-tuned (GAP)"),
               ("r50_ft_head", "ResNet50, fine-tuned (head, 256-d)"),
               ("cnn_head", "Custom CNN (head, 128-d)")]
    rows = []
    for tag, name in sources:
        p = out("emb", f"{tag}.npy")
        if not os.path.exists(p):
            continue
        X = StandardScaler().fit_transform(np.load(p))
        Z = PCA(n_components=min(CFG["cluster_pca_dims"], X.shape[1]), random_state=0).fit_transform(X)
        sil_true = float(silhouette_score(Z, y))
        for method in ("spectral", "kmeans"):
            m = {k: [] for k in ("sil", "ari", "nmi", "ami")}
            for s in CFG["cluster_seeds"]:
                if method == "spectral":
                    lab = SpectralClustering(n_clusters=10, affinity="nearest_neighbors",
                                             n_neighbors=CFG["cluster_n_neighbors"],
                                             assign_labels="cluster_qr", random_state=s).fit_predict(Z)
                else:
                    lab = KMeans(n_clusters=10, n_init=10, random_state=s).fit_predict(Z)
                m["sil"].append(silhouette_score(Z, lab) if len(set(lab)) > 1 else 0.0)
                m["ari"].append(adjusted_rand_score(y, lab))
                m["nmi"].append(normalized_mutual_info_score(y, lab))
                m["ami"].append(adjusted_mutual_info_score(y, lab))
            rows.append(dict(tag=tag, name=name, method=method, sil_true=sil_true,
                             **{k: v for k, v in m.items()}, Z2=Z[:, :2]))
    return rows, y


def stage_report(kaggle_private=None, kaggle_public=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = load_runs()
    df = get_split()
    classes = runs[CONFIGS[0]][0]["classes"]
    seeds = sorted({r["seed"] for rs in runs.values() for r in rs})
    best, runner = select_best(runs)
    M = {}   # LaTeX macros

    M["NTrain"] = f"{(df.split == 'train').sum():,}"
    M["NVal"] = f"{(df.split == 'val').sum():,}"
    M["NTest"] = f"{(df.split == 'test').sum():,}"
    M["NSeeds"] = str(len(seeds))

    # ---- Table IV: main comparison
    lines = []
    for c in CONFIGS:
        rs = runs.get(c, [])
        if not rs:
            continue
        gap = [r["history"]["accuracy"][best_epoch(r["history"])] - r["history"]["val_accuracy"][best_epoch(r["history"])] for r in rs]
        ep = [len(r["history"]["loss"]) for r in rs]
        name = PRETTY[c] + (r"$^{\ast}$" if c == best else "")
        lines.append(" & ".join([name, pm([r["val"]["macro_f1"] for r in rs]),
                                 pm([r["test"]["acc"] for r in rs]),
                                 pm([r["test"]["macro_f1"] for r in rs]),
                                 pm([r["test"]["weighted_f1"] for r in rs]),
                                 pm([r["test"]["loss"] for r in rs], pct=False, nd=3),
                                 pm(gap), f"{np.mean(ep):.1f}"]) + r" \\")
    open(out("tab_main.tex"), "w").write("\n".join(lines) + "\n")

    acc = lambda c: [r["test"]["acc"] for r in runs[c]]
    for key, c in (("CNNAcc", "cnn"), ("MNVFrozenAcc", "mnv2_frozen"), ("MNVFtAcc", "mnv2_ft"),
                   ("ResFrozenAcc", "r50_frozen"), ("ResFtAcc", "r50_ft")):
        M[key] = pm(acc(c)) + r"\%"
    M["DeltaMNV"] = f"{100 * (np.mean(acc('mnv2_ft')) - np.mean(acc('mnv2_frozen'))):+.2f}"
    M["DeltaRes"] = f"{100 * (np.mean(acc('r50_ft')) - np.mean(acc('r50_frozen'))):+.2f}"
    M["BestModel"] = PRETTY[best]
    M["RunnerUp"] = PRETTY[runner]
    M["BestTestAcc"] = pm(acc(best)) + r"\%"
    M["BestTestAccShort"] = mean_s(acc(best)) + r"\%"
    M["BestMacroF"] = pm([r["test"]["macro_f1"] for r in runs[best]]) + r"\%"
    M["BestWeightedF"] = pm([r["test"]["weighted_f1"] for r in runs[best]]) + r"\%"
    bh = [r["history"] for r in runs[best]]
    M["BestGap"] = mean_s([h["accuracy"][best_epoch(h)] - h["val_accuracy"][best_epoch(h)] for h in bh])

    # McNemar per seed, best vs runner-up, on the shared test split
    yte = df[df.split == "test"].y.values
    pv = []
    for rb in runs[best]:
        rr = next((r for r in runs[runner] if r["seed"] == rb["seed"]), None)
        if rr:
            pv.append(mcnemar(rb["test"]["pred"], rr["test"]["pred"], yte))
    M["McNemarMaxP"] = f"{max(pv):.3g}" if pv else "n/a"
    M["McNemarSig"] = f"{sum(p < 0.05 for p in pv)} of {len(pv)}"

    # ---- Table V: per-class (best config, mean over seeds)
    P = np.array([r["test"]["precision"] for r in runs[best]])
    R = np.array([r["test"]["recall"] for r in runs[best]])
    F = np.array([r["test"]["f1"] for r in runs[best]])
    S = runs[best][0]["test"]["support"]
    lines = []
    for k, c in enumerate(classes):
        lines.append(f"{CLASS_PRETTY.get(c, c)} & {S[k]} & {100*P[:,k].mean():.2f} & "
                     f"{100*R[:,k].mean():.2f} & {100*F[:,k].mean():.2f} $\\pm$ {100*F[:,k].std(ddof=1) if len(F)>1 else 0:.2f} \\\\")
    open(out("tab_perclass.tex"), "w").write("\n".join(lines) + "\n")
    worst = int(np.argmin(F.mean(0)))
    M["WorstClass"] = CLASS_PRETTY.get(classes[worst], classes[worst]).lower()
    M["WorstClassF"] = f"{100 * F[:, worst].mean():.2f}\\%"
    M["MinorityRecall"] = f"{100 * R[:, classes.index('bacterial_panicle_blight')].mean():.2f}\\%"

    # ---- Fig: learning curves (first seed)
    s0 = seeds[0]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for c in CONFIGS:
        r = next((x for x in runs.get(c, []) if x["seed"] == s0), None)
        if r is None:
            continue
        e = np.arange(1, len(r["history"]["val_accuracy"]) + 1)
        ax[0].plot(e, r["history"]["val_accuracy"], label=PRETTY[c])
        if r["phase_boundary"]:
            ax[0].axvline(r["phase_boundary"] + 0.5, color="grey", lw=0.6, ls=":")
    ax[0].set(xlabel="Epoch", ylabel="Validation accuracy", title="Validation accuracy (seed %d)" % s0)
    ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
    r = next(x for x in runs[best] if x["seed"] == s0)
    e = np.arange(1, len(r["history"]["loss"]) + 1)
    ax[1].plot(e, r["history"]["loss"], label="Training loss")
    ax[1].plot(e, r["history"]["val_loss"], label="Validation loss")
    if r["phase_boundary"]:
        ax[1].axvline(r["phase_boundary"] + 0.5, color="grey", lw=0.8, ls=":", label="Unfreeze")
    ax[1].set(xlabel="Epoch", ylabel="Cross-entropy", title=f"{PRETTY[best]} loss (seed {s0})")
    ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out("fig_curves.png"), dpi=200); plt.close(fig)

    # ---- Fig: confusion matrix (best config, summed over seeds, row-normalised)
    cm = np.sum([np.array(r["test"]["cm"]) for r in runs[best]], axis=0).astype(float)
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    names = [CLASS_PRETTY.get(c, c) for c in classes]
    ax.set_xticks(range(10)); ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(10)); ax.set_yticklabels(names, fontsize=7)
    for i in range(10):
        for j in range(10):
            if cmn[i, j] >= 0.01:
                ax.text(j, i, f"{cmn[i,j]:.2f}", ha="center", va="center", fontsize=5.5,
                        color="white" if cmn[i, j] > .5 else "black")
    ax.set(xlabel="Predicted", ylabel="True")
    fig.colorbar(im, fraction=0.046); fig.tight_layout()
    fig.savefig(out("fig_confusion.png"), dpi=220); plt.close(fig)

    # ---- Representation analysis
    rows, y = run_clustering()
    lines = []
    for row in rows:
        lines.append(" & ".join([row["name"], row["method"].capitalize(), pm(row["sil"], pct=False, nd=3),
                                 pm(row["ari"], pct=False, nd=3), pm(row["nmi"], pct=False, nd=3),
                                 pm(row["ami"], pct=False, nd=3), f"{row['sil_true']:.3f}"]) + r" \\")
    open(out("tab_cluster.tex"), "w").write("\n".join(lines) + "\n")
    get = lambda tag, met, k: next((r[k] for r in rows if r["tag"] == tag and r["method"] == met), None)
    for key, tag in (("Rand", "r50_random_gap"), ("Img", "r50_imagenet_gap"), ("Ft", "r50_ft_gap"),
                     ("FtHead", "r50_ft_head"), ("CNN", "cnn_head")):
        for met in ("ami", "ari", "nmi", "sil"):
            v = get(tag, "spectral", met)
            if v is not None:
                M[f"Clu{key}{met.upper()}"] = f"{np.mean(v):.3f}"
    sel = [r for r in rows if r["method"] == "spectral" and r["tag"] in ("r50_imagenet_gap", "r50_ft_gap")]
    if len(sel) == 2:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        for a, r in zip(ax, sel):
            sc = a.scatter(r["Z2"][:, 0], r["Z2"][:, 1], c=y, cmap="tab10", s=5, alpha=.7)
            a.set(title=r["name"] + " - true labels", xlabel="PC1", ylabel="PC2")
        handles = [plt.Line2D([], [], marker="o", ls="", color=plt.cm.tab10(k / 9), label=CLASS_PRETTY.get(c, c))
                   for k, c in enumerate(classes)]
        fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=7)
        fig.tight_layout(rect=(0, .1, 1, 1)); fig.savefig(out("fig_embeddings.png"), dpi=200); plt.close(fig)

    # ---- Kaggle independent test
    if kaggle_private is not None:
        M["KagglePrivate"] = f"{100 * kaggle_private:.2f}\\%"
    if kaggle_public is not None:
        M["KagglePublic"] = f"{100 * kaggle_public:.2f}\\%"

    with open(out("results.tex"), "w") as f:
        f.write("% auto-generated by run_experiments.py -- do not edit by hand\n")
        for k, v in M.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    json.dump({k: v for k, v in M.items()}, open(out("results_summary.json"), "w"), indent=1)
    print("Selected (validation macro-F1):", best, "| runner-up:", runner)
    print("Wrote results.tex, tab_main.tex, tab_perclass.tex, tab_cluster.tex and figures to", CFG["out_dir"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["train", "embed", "submit", "report", "all"], default="all")
    ap.add_argument("--data_dir"); ap.add_argument("--out_dir")
    ap.add_argument("--kaggle_private", type=float); ap.add_argument("--kaggle_public", type=float)
    ap.add_argument("--seeds", help="comma-separated, e.g. 0,1,2")
    a = ap.parse_args()
    if a.seeds: CFG["seeds"] = [int(x) for x in a.seeds.split(",")]
    if a.data_dir: CFG["data_dir"] = a.data_dir
    if a.out_dir: CFG["out_dir"] = a.out_dir
    json.dump(CFG, open(out("config_used.json"), "w"), indent=1)
    if a.stage in ("train", "all"): stage_train()
    if a.stage in ("embed", "all"): stage_embed()
    if a.stage in ("submit", "all"): stage_submit()
    if a.stage in ("report", "all"): stage_report(a.kaggle_private, a.kaggle_public)
