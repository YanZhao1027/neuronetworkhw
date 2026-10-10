#!/usr/bin/env python3
"""SCUT deep learning 2026 / Experiment 1. Reproducible end-to-end run.

Required tasks: NumPy, sine linear regression, cubic regression, MNIST softmax.
Clearly separated extensions: finite-difference gradient test, polynomial
generalization, confusion/error/confidence analysis, optional convolutional net.
No fabricated numerical results; every artifact is generated from computations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torchvision.datasets import MNIST

BASE = Path(__file__).resolve().parent
OUT = BASE / "results"
FIG = BASE / "figures"
DATA = BASE / "data"
SEED = 2026

plt.rcParams.update({
    "figure.dpi": 115, "savefig.dpi": 190, "font.size": 11,
    "axes.titlesize": 13, "axes.labelsize": 11, "figure.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
})


def save_plot(name: str):
    FIG.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIG / name, bbox_inches="tight", facecolor="white")
    plt.close()


def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def append_log(msg):
    print(msg, flush=True)
    with (OUT / "training.log").open("a", encoding="utf-8") as fp:
        fp.write(str(msg) + "\n")


def run_numpy():
    result = subprocess.run([sys.executable, str(BASE / "numpy_exercise.py")],
                            text=True, capture_output=True, check=True)
    (OUT / "numpy_output.txt").write_text(result.stdout, encoding="utf-8")
    append_log("NumPy checks done; all code and genuine output in numpy_exercise.py / results/numpy_output.txt")
    create_numpy_evidence(result.stdout)


def create_numpy_evidence(out):
    """Two readable code/output panels, based on exact source and stdout."""
    from PIL import Image, ImageDraw, ImageFont
    src = (BASE / "numpy_exercise.py").read_text(encoding="utf-8").splitlines()
    lines = out.splitlines()
    font_file = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
    candidates = [font_file, "/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf"]
    ff = next((p for p in candidates if Path(p).exists()), None)
    font = ImageFont.truetype(ff, 23) if ff else ImageFont.load_default()
    small = ImageFont.truetype(ff, 20) if ff else ImageFont.load_default()
    width = 2600
    cw = 1260
    pad = 50
    line_height = 32
    height = 170 + line_height * max(len(src), len(lines))
    im = Image.new("RGB", (width, height), "#111827")
    draw = ImageDraw.Draw(im)
    draw.text((55, 25), "NUMPY EXERCISE / SOURCE", font=font, fill="#82d5ff")
    draw.text((1350, 25), "GENUINE PROGRAM OUTPUT", font=font, fill="#82d5ff")
    draw.line((width//2, 25, width//2, height-25), fill="#374151", width=3)
    for i, line in enumerate(src):
        draw.text((pad, 95 + i*line_height), line[:96], font=small, fill="#ebecf0")
    for i, line in enumerate(lines):
        draw.text((1350, 95 + i*line_height), line[:92], font=small, fill="#b6ffcd")
    im.save(FIG / "01_numpy_code_and_output.png")


def regression_data():
    rng = np.random.default_rng(SEED)
    x = np.linspace(-3.0, 3.0, 100, dtype=np.float32)
    eps = rng.uniform(-0.5, 0.5, size=100).astype(np.float32)
    y = (np.sin(x) + eps).astype(np.float32)
    np.savetxt(OUT / "sine_training_data.csv", np.column_stack([x, y, eps]),
               delimiter=",", header="x,y,uniform_noise", comments="")
    plt.figure(figsize=(8.2, 4.6))
    plt.scatter(x, y, s=24, alpha=.8, color="#3377b9", label="100 noisy samples")
    plt.plot(x, np.sin(x), "--", color="#888888", lw=1.7, label="sin(x) (unobserved truth)")
    plt.title("Sine regression dataset: 100 samples, uniform noise [-0.5, 0.5]")
    plt.xlabel("x"); plt.ylabel("y"); plt.legend()
    save_plot("02_sine_scatter.png")
    return x, y, eps


def fit_regression(x, y, degree=1):
    """MSE full-batch gradient optimization, CPU is faster for n=100."""
    seed_all(SEED + degree)
    features = np.column_stack([x**p for p in range(1, degree+1)]).astype(np.float32)
    xt = torch.from_numpy(features)
    yt = torch.from_numpy(y[:, None].astype(np.float32))
    model = nn.Linear(degree, 1)
    optim = torch.optim.Adam(model.parameters(), lr=0.03)
    criterion = nn.MSELoss()
    history = []
    steps = 1500 if degree == 1 else 2600
    start = time.perf_counter()
    for step in range(steps):
        optim.zero_grad(set_to_none=True)
        loss = criterion(model(xt), yt)
        loss.backward()
        optim.step()
        if step % 10 == 0 or step == steps-1:
            history.append((step + 1, float(loss.item())))
    with torch.inference_mode():
        pred = model(xt).numpy().ravel()
        true_mse = float(criterion(model(xt), yt).item())
    weights = [float(t) for t in model.weight.detach().flatten()]
    bias = float(model.bias.detach().item())
    return {"degree":degree, "weights":weights, "bias":bias,
            "mse":true_mse, "train_seconds":round(time.perf_counter()-start, 3),
            "steps":steps, "loss_history":history}, pred


def gradient_check():
    """Numerical differentiation with double precision, separate from baseline."""
    xx = torch.tensor([-1.2, .5, 1.8], dtype=torch.float64)
    yy = torch.tensor([-.7, .4, 1.1], dtype=torch.float64)
    w = torch.tensor(.37, requires_grad=True, dtype=torch.float64)
    b = torch.tensor(-.21, requires_grad=True, dtype=torch.float64)
    loss = ((w * xx + b - yy)**2).mean()
    loss.backward()
    h = 1e-5
    def f(wv): return float(((wv*xx + b.detach()-yy)**2).mean())
    finite = (f(w.detach()+h)-f(w.detach()-h))/(2*h)
    autodiff = float(w.grad)
    return {"autograd_dw":autodiff, "finite_difference_dw":finite,
            "absolute_error":abs(autodiff-finite),
            "relative_error":abs(autodiff-finite)/max(abs(autodiff), 1e-12)}


def polynomial_extrapolation(x, y):
    """Extension: polynomial interpolation/extrapolation; NOT NN training."""
    inside = np.linspace(-3, 3, 401)
    outside = np.linspace(-4.5, 4.5, 601)
    xp = np.linspace(-4.5, 4.5, 601)
    result = {}
    plt.figure(figsize=(10, 5.6))
    plt.plot(xp, np.sin(xp), color="#151923", lw=2.4, label="sin(x), true")
    palette = {1:"#e0983b", 3:"#277ca8", 5:"#46a28e", 9:"#bd5c7e"}
    for deg in [1, 3, 5, 9]:
        coef = np.polynomial.polynomial.polyfit(x, y, deg=deg)
        inhat = np.polynomial.polynomial.polyval(inside, coef)
        outhat = np.polynomial.polynomial.polyval(outside, coef)
        trhat = np.polynomial.polynomial.polyval(x, coef)
        result[str(deg)] = {
            "training_mse_noisy":float(np.mean((trhat-y)**2)),
            "interpolation_clean_mse":float(np.mean((inhat-np.sin(inside))**2)),
            "extrapolation_clean_mse":float(np.mean((outhat[np.abs(outside)>3] -
                                                   np.sin(outside[np.abs(outside)>3]))**2))}
        pred = np.polynomial.polynomial.polyval(xp, coef)
        plt.plot(xp, np.clip(pred, -8, 8), lw=1.8, label=f"degree {deg}", color=palette[deg])
    plt.scatter(x, y, s=8, color="gray", alpha=.22, zorder=5)
    plt.axvline(-3, color="gray", ls=":", lw=1.4)
    plt.axvline(3, color="gray", ls=":", lw=1.4)
    plt.ylim(-3.5, 3.5)
    plt.xlabel("x (outside dashed boundaries = extrapolation)")
    plt.ylabel("prediction")
    plt.title("Extension: flexibility helps interpolation, can hurt extrapolation")
    plt.legend(ncol=3)
    save_plot("06_polynomial_extrapolation.png")
    return result


def run_regression():
    x, y, eps = regression_data()
    linear, pred1 = fit_regression(x, y, degree=1)
    cubic, pred3 = fit_regression(x, y, degree=3)
    for result, pred, name, color in [
        (linear, pred1, "03_linear_regression.png", "#cc6d48"),
        (cubic, pred3, "04_cubic_regression.png", "#1f9a87"),
    ]:
        plt.figure(figsize=(8.2, 4.7))
        plt.scatter(x, y, label="Training data", s=23, alpha=.7, color="#4483c0")
        plt.plot(x, pred, color=color, linewidth=2.7, label=f"Fitted degree {result['degree']}")
        plt.plot(x, np.sin(x), color="#353d49", ls="--", lw=1.4, label="sin(x)")
        plt.xlabel("x"); plt.ylabel("y")
        plt.title(f"Degree {result['degree']} regression | training MSE = {result['mse']:.5f}")
        plt.legend()
        save_plot(name)
    plt.figure(figsize=(8.8, 4.7))
    plt.scatter(x, y, s=18, alpha=.55, label="Observed samples", color="#9a9a9a")
    plt.plot(x, np.sin(x), ls="--", color="black", label="True sin(x)")
    plt.plot(x, pred1, lw=2.2, color="#cc6d48",
             label=f"Linear MSE={linear['mse']:.4f}")
    plt.plot(x, pred3, lw=2.2, color="#1f9a87",
             label=f"Cubic MSE={cubic['mse']:.4f}")
    plt.xlabel("x"); plt.ylabel("y");plt.legend()
    plt.title("Same training data, different model capacity")
    save_plot("05_linear_vs_cubic.png")
    plt.figure(figsize=(8.4, 4.6))
    for result, color in [(linear, "#cc6d48"), (cubic, "#1f9a87")]:
        h = np.array(result["loss_history"])
        plt.plot(h[:,0], h[:,1], color=color, lw=2,
                 label=f"Degree {result['degree']}")
    plt.yscale("log");plt.xlabel("Gradient update step");plt.ylabel("MSE (log scale)")
    plt.title("Regression optimization histories");plt.legend()
    save_plot("07_regression_training_loss.png")
    extension = polynomial_extrapolation(x, y)
    grad = gradient_check()
    append_log(f"Linear regression: w={linear['weights'][0]:.6f}; b={linear['bias']:.6f}; MSE={linear['mse']:.6f}")
    append_log(f"Cubic regression: weights={cubic['weights']}; b={cubic['bias']:.6f}; MSE={cubic['mse']:.6f}")
    append_log(f"Gradient check: numerical-vs-autograd relative error={grad['relative_error']:.2e}")
    return {"samples":100,"noise_distribution":"Uniform(-0.5, 0.5)",
            "x_range":[-3,3], "noise_sample_mean":float(eps.mean()),
            "theoretical_uniform_noise_variance":1/12,
            "linear":linear,"cubic":cubic,"gradient_check":grad,
            "polynomial_extrapolation":extension}


class SoftmaxMNIST(nn.Module):
    """Exactly the assigned 784-to-10 linear classifier. Returns LOGITS."""
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        return self.fc(self.flatten(x))


class CNNExtension(nn.Module):
    """Not part of required model. Kept separate for an optional comparison."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32*7*7, 64), nn.ReLU(), nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.net(x)


def get_mnist(device):
    root = DATA / "mnist"
    append_log("Acquiring authentic torchvision MNIST dataset (60000 train / 10000 test)...")
    tr = MNIST(root=str(root), train=True, download=True)
    te = MNIST(root=str(root), train=False, download=True)
    assert len(tr) == 60000 and len(te) == 10000
    train_x = tr.data.unsqueeze(1).float().div_(255.0).to(device)
    test_x = te.data.unsqueeze(1).float().div_(255.0).to(device)
    train_y = tr.targets.long().to(device)
    test_y = te.targets.long().to(device)
    append_log(f"X_train={tuple(train_x.shape)}, Y_train={tuple(train_y.shape)}, "
               f"X_test={tuple(test_x.shape)}; on {device}")
    return train_x, train_y, test_x, test_y


@torch.inference_mode()
def evaluate(model, xx, yy, batch=1024):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss, correct = 0.0, 0
    logits_list = []
    for start in range(0, len(xx), batch):
        logits = model(xx[start:start+batch])
        labels = yy[start:start+batch]
        total_loss += criterion(logits, labels).item()
        correct += (logits.argmax(dim=1) == labels).sum().item()
        logits_list.append(logits.cpu())
    return ({"loss":float(total_loss / len(yy)),
             "accuracy":float(correct / len(yy))},
            torch.cat(logits_list, dim=0))


def train_mnist(model, data, device, epochs=10, batch_size=256, lr=.2, method="SGD"):
    tr_x, tr_y, te_x, te_y = data
    model = model.to(device)
    optimizer = (torch.optim.SGD(model.parameters(), lr=lr) if method == "SGD" else
                 torch.optim.Adam(model.parameters(), lr=lr))
    criterion = nn.CrossEntropyLoss()
    history = []
    steps = 0
    tic = time.perf_counter()
    for ep in range(epochs):
        model.train()
        idx = torch.randperm(len(tr_x), device=device)
        for ids in idx.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            logits = model(tr_x[ids])
            loss = criterion(logits, tr_y[ids])
            loss.backward()
            optimizer.step()
            steps += 1
        tr_metrics, _ = evaluate(model, tr_x, tr_y)
        te_metrics, _ = evaluate(model, te_x, te_y)
        history.append({"epoch":ep+1, "iteration":steps,
                        "train_loss":tr_metrics["loss"],
                        "train_accuracy":tr_metrics["accuracy"],
                        "test_loss":te_metrics["loss"],
                        "test_accuracy":te_metrics["accuracy"]})
        append_log(f"{model.__class__.__name__} epoch {ep+1:02d}/{epochs} "
                   f"step {steps:4d} train loss {tr_metrics['loss']:.4f} "
                   f"train acc {tr_metrics['accuracy']:.4%} "
                   f"test loss {te_metrics['loss']:.4f} "
                   f"test acc {te_metrics['accuracy']:.4%}")
    secs = time.perf_counter()-tic
    te_metrics, logits = evaluate(model, te_x, te_y)
    torch.save(model.state_dict(), OUT / ("softmax_weights.pt" if method=="SGD" else "cnn_extension_weights.pt"))
    return {"model":model.__class__.__name__,
            "parameter_count":sum(p.numel() for p in model.parameters()),
            "train_examples":len(tr_x), "test_examples":len(te_x),
            "optimizer":method, "learning_rate":lr,
            "batch_size":batch_size, "epochs":epochs,
            "wall_seconds":round(secs, 3),
            "history":history, "final_test":te_metrics}, logits


def plot_history(records, prefix, name):
    xx = [r["iteration"] for r in records]
    for metric, file_name in [("accuracy", f"{prefix}_accuracy.png"),
                              ("loss", f"{prefix}_loss.png")]:
        plt.figure(figsize=(9.1, 5))
        y_train=[r[f"train_{metric}"] for r in records]
        y_test=[r[f"test_{metric}"] for r in records]
        if metric=="accuracy":
            y_train=np.array(y_train)*100
            y_test=np.array(y_test)*100
        plt.plot(xx, y_train, "-o", lw=2.2, label="Train", color="#236d95")
        plt.plot(xx, y_test, "-s", lw=2.2, label="Test", color="#d27440")
        plt.xlabel("Cumulative parameter updates (iterations)")
        plt.ylabel("Accuracy (%)" if metric=="accuracy" else "Mean cross-entropy")
        plt.title(f"{name}: train vs test {metric}")
        plt.legend();plt.grid(alpha=.13)
        save_plot(file_name)


def plot_errors(y_true, logits, model):
    pred = logits.argmax(dim=1)
    conf = logits.softmax(dim=1).max(dim=1).values
    target = y_true.cpu().long()
    confusion = torch.bincount(target*10 + pred, minlength=100).reshape(10,10).numpy()
    np.savetxt(OUT / "softmax_confusion_matrix.csv", confusion, delimiter=",",fmt="%d")
    class_counts = confusion.sum(axis=1)
    per_class_acc = (np.diag(confusion)/class_counts).tolist()
    cm_rate = confusion / np.maximum(class_counts[:,None],1)
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    im=ax.imshow(cm_rate, cmap="Blues",vmin=0,vmax=1)
    fig.colorbar(im, ax=ax, label="Within-true-class proportion")
    ax.set_xticks(range(10));ax.set_yticks(range(10))
    ax.set_xlabel("Predicted digit");ax.set_ylabel("True digit")
    ax.set_title("Extension: normalized confusion matrix | softmax")
    for i in range(10):
        for j in range(10):
            if cm_rate[i,j] > .03 or i==j:
                ax.text(j,i,f"{cm_rate[i,j]:.2f}",ha="center",va="center",
                        fontsize=8,color="white" if cm_rate[i,j]>.56 else "#1f2833")
    save_plot("10_softmax_confusion_matrix.png")
    top_pairs=[]
    for i in range(10):
        for j in range(10):
            if i != j: top_pairs.append((int(confusion[i,j]),i,j))
    top_pairs.sort(reverse=True)
    wrong = (pred != target).nonzero().flatten()
    hardest = wrong[torch.argsort(conf[wrong], descending=True)[:20]]
    test_data = MNIST(root=str(DATA/"mnist"),train=False,download=False)
    fig, axes=plt.subplots(4,5,figsize=(10.8,8.4))
    for ax,index in zip(axes.flat,hardest.tolist()):
        ax.imshow(test_data.data[index].numpy(),cmap="gray_r",vmin=0,vmax=255)
        ax.set_title(f"true {int(target[index])} / pred {int(pred[index])}\nconf {float(conf[index]):.2f}",
                     fontsize=10,color="#b34c48")
        ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle("Extension: 20 most confident mistakes (model limitations)",fontsize=15)
    save_plot("11_softmax_high_confidence_errors.png")
    # Reliability calibration diagram (10 bins).
    edges = np.linspace(0,1,11)
    confidences = conf.numpy()
    correct=(pred==target).float().numpy()
    acc_bins, conf_bins, counts = [],[],[]
    ece=0.0
    for bi in range(10):
        mask=(confidences>=edges[bi]) & (confidences<edges[bi+1] if bi<9 else confidences<=1)
        c=int(mask.sum())
        counts.append(c)
        if c:
            a=float(correct[mask].mean());p=float(confidences[mask].mean())
            ece+=c/len(correct)*abs(a-p)
        else:a=0.;p=0.
        acc_bins.append(a);conf_bins.append(p)
    plt.figure(figsize=(7.6,5.3))
    plt.plot([0,1],[0,1],"--",color="gray",label="Perfectly calibrated")
    values=[(a,p,c) for a,p,c in zip(acc_bins,conf_bins,counts) if c]
    plt.scatter([p for a,p,c in values],[a for a,p,c in values],
                s=[max(30,c/20) for a,p,c in values],alpha=.75,color="#287c9e",
                label="Confidence bins (size=count)")
    plt.xlim(0,1);plt.ylim(0,1);plt.xlabel("Mean confidence")
    plt.ylabel("Observed fraction correct")
    plt.title(f"Extension: softmax reliability | ECE={ece:.4f}")
    plt.legend()
    save_plot("12_softmax_calibration.png")
    return {"per_class_accuracy":per_class_acc,
            "top_confusions":[{"count":x,"true":i,"predicted":j} for x,i,j in top_pairs[:8]],
            "wrong_count":int(len(wrong)), "ece_10_bins":float(ece),
            "confidence_bins":[{"count":int(c),"mean_confidence":p,"accuracy":a}
                               for c,p,a in zip(counts,conf_bins,acc_bins)]}


def plot_digit_weights(model):
    # Layer filters remain signed linear pixel coefficients (NOT sample digits).
    weights = model.fc.weight.detach().cpu().numpy().reshape(10,28,28)
    lim=float(np.max(np.abs(weights)))
    fig, axes=plt.subplots(2,5,figsize=(11,5.6))
    for i,ax in enumerate(axes.flat):
        ax.imshow(weights[i],cmap="RdBu_r",vmin=-lim,vmax=lim)
        ax.set_title(f"Class {i} weight map",fontsize=10)
        ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle("Extension: what pixels increase/decrease each digit logit",fontsize=14)
    save_plot("13_softmax_weight_maps.png")


def run_mnist(device, cnn_epochs=3, skip_cnn=False):
    seed_all(SEED)
    data=get_mnist(device)
    baseline=SoftmaxMNIST()
    base, logit=train_mnist(baseline,data,device,epochs=10,batch_size=256,lr=.2,method="SGD")
    plot_history(base["history"],"08_softmax","Required 784-to-10 MNIST softmax")
    diagnostics=plot_errors(data[3],logit,baseline)
    plot_digit_weights(baseline)
    append_log(f"Softmax: test accuracy={base['final_test']['accuracy']:.4%}, "
               f"test CE={base['final_test']['loss']:.5f}, "
               f"parameters={base['parameter_count']}, ECE={diagnostics['ece_10_bins']:.5f}")
    output={"softmax":base, "softmax_diagnostics":diagnostics}
    if not skip_cnn:
        seed_all(SEED)
        cnn=CNNExtension()
        comparison,_=train_mnist(cnn,data,device,epochs=cnn_epochs,
                                 batch_size=256,lr=.001,method="Adam")
        plot_history(comparison["history"],"14_cnn_extension","Optional CNN extension")
        output["cnn_extension"]=comparison
        plt.figure(figsize=(8.3,4.7))
        labels=["Linear softmax","CNN extension"]
        accuracies=[base["final_test"]["accuracy"]*100,comparison["final_test"]["accuracy"]*100]
        bars=plt.bar(labels,accuracies,width=.46,color=["#3274a1","#3da48c"])
        plt.ylim(85,100)
        for rect,val in zip(bars,accuracies):
            plt.text(rect.get_x()+rect.get_width()/2,val+.12,f"{val:.2f}%",
                     ha="center",fontsize=12,fontweight="bold")
        plt.ylabel("MNIST test accuracy (%)")
        plt.title("Extension: inductive bias matters (not the required architecture)")
        save_plot("15_softmax_vs_cnn_extension.png")
        append_log(f"Separate CNN extension: test accuracy={comparison['final_test']['accuracy']:.4%}, "
                   f"test CE={comparison['final_test']['loss']:.5f}, "
                   f"parameters={comparison['parameter_count']}")
    return output


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--skip-cnn",action="store_true",help="Only run required tasks and diagnostic extensions")
    parser.add_argument("--cnn-epochs",type=int,default=3)
    parser.add_argument("--device",choices=["auto","cpu","cuda"],default="auto")
    parser.add_argument("--only",choices=["all","numpy","regression","mnist"],default="all")
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True); DATA.mkdir(exist_ok=True)
    seed_all()
    device=("cuda" if torch.cuda.is_available() else "cpu") if args.device=="auto" else args.device
    if device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    metrics={
        "experiment":"SCUT neural network experiment 1 (2026)",
        "seed":SEED,"run_timestamp":datetime.now().astimezone().isoformat(),
        "environment":{"python":platform.python_version(),"torch":torch.__version__,
                       "torchvision":"see environment file","cuda_build":torch.version.cuda,
                       "cuda_available":torch.cuda.is_available(),
                       "device":device, "gpu":torch.cuda.get_device_name(0) if device=="cuda" else None,
                       "OS":platform.platform()},
        "scientific_note":"Regression MSE measured on training samples; MNIST test values measured on genuine held-out test set. Separate CNN is extension only.",
    }
    (OUT/"training.log").write_text("",encoding="utf-8")
    append_log(f"START | seed={SEED} | device={device} | torch={torch.__version__}")
    if args.only in ("all","numpy"):run_numpy()
    if args.only in ("all","regression"):metrics["regression"]=run_regression()
    if args.only in ("all","mnist"):metrics["mnist"]=run_mnist(device,args.cnn_epochs,args.skip_cnn)
    with (OUT/"metrics.json").open("w",encoding="utf-8") as f:
        json.dump(metrics,f,ensure_ascii=False,indent=2)
    append_log("SUCCESS: saved results/metrics.json and figures")
    print("\nRESULTS_JSON",OUT/"metrics.json")


if __name__=="__main__":
    main()
