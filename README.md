# 2026 神经网络与深度学习实验

RTX 5090 / PyTorch 实验记录：真实数据、源码、训练结果与可在线编辑的报告。

## 实验导航

| 实验 | 内容 | 状态 |
|---|---|---|
| [实验一](experiments/exp1/) | NumPy、线性回归、三次回归、MNIST Softmax 和 CNN 拓展 | 已完成 |
| [实验二](experiments/exp2/) | 后续作业 | 待发布 |
| [实验三](experiments/exp3/) | 后续作业 | 待发布 |

## 实验一：快速入口

- [网页直接编辑 Markdown 报告](experiments/exp1/report.md)
- [下载 Word DOCX 初始模板](experiments/exp1/report-template.docx)
- [完整代码](experiments/exp1/run_all.py)
- [可视化图表](experiments/exp1/figures/)
- [真实训练指标](experiments/exp1/results/metrics.json)

MNIST Softmax 测试精度：92.07%；独立 CNN 拓展：97.72%。报告所有数字均取自实际训练。

## 其他设备编辑 Word

**方案一**：直接下载 report-template.docx，用任意设备的 Microsoft Word 或 WPS 修改。这是二进制文件，GitHub 网页不能直接编辑其正文；编辑后需另行推送文件。

**方案二（建议）**：打开 report.md，点击 GitHub 的编辑铅笔并提交；本项目 GitHub Actions 的 Build editable Word 工作流自动将 Markdown 导出 DOCX。前往 Actions 找到最新运行，在 Artifacts 下载 exp1-word-from-markdown 压缩包，然后解压得到 Word。网页改 Markdown 和本地改 Word 是两套编辑来源，建议以 report.md 为今后的跨设备主文稿。

## 本机运行

安装兼容你的 GPU 的 PyTorch、torchvision 及 numpy、matplotlib、pillow、python-docx。

    cd experiments/exp1
    python run_all.py --device auto
    python create_report.py

可以添加 --skip-cnn 跳过非作业必做 CNN。首次运行会下载 MNIST 数据集。生成图片存放在 figures，指标存放在 results。

## 隐私与后续作业

这是**公开仓库**。禁止将个人姓名、学号、班级或课程平台凭证填入公开文档。正式提交 DOCX 请保存到本地 submission 文件夹（已被 gitignore 排除）。后续作业按 experiments/expN 分目录维护，不提前编造新实验结果。

课程原始要求：https://github.com/caixd-220529/deel_learning_2026-public/blob/main/experiments/exp1/实验一.md
