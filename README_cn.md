# Flow Inversion Img2Img (ComfyUI) — 已归档

[English](README.md) | [Русский](README_ru.md) | **中文**

> **状态：项目已关闭，研究目标未能实现。**
> 如果需要在 flow matching 模型上进行基于文本的图像编辑，请使用 **[FlowEdit](https://arxiv.org/abs/2412.08629)**，而不是本仓库。

## 项目的初衷

本项目原本希望做一项完整的研究：对 Flow Matching（Anima、Wan2.1、Flux、SD3）的 ODE 进行数学上修正的反演，使其能够在 ComfyUI 中无需蒙版进行 Img2Img 编辑，同时保留细节（线稿、手指、衣物褶皱、背景）。

## 实际结果

代码会记录反演轨迹并逐步回放（`x ← x − d_k`）。因此重建是**构造上精确**的，而不是因为证明了 ODE 可逆。在回放之上，再叠加类似 FlowEdit 的差分项 `v_target − v_source`。这在本质上与 FlowEdit 是同一个思路，因此本项目相对于 FlowEdit 并没有实质性的创新。

经过四天的实验（单一模型、单张图像、单个 seed），得到以下结论：

* 小幅编辑可行：发色和瞳色可以改变，结构得以保留。
* 大面积区域（例如白色连衣裙变为黑色）无法重新着色。源潜变量的结构会固定颜色，加大编辑强度得到的不是目标颜色，而是画面崩坏。
* 编辑强度过大时会出现“水彩”效果：发光的光晕和模糊的形体。这是一种伪影（CFG 过饱和 / 偏离流形崩塌），无法控制，没有任何实际用途。
* 发现了若干错误和异常（σ 索引错误、两个分支的 CFG 不一致、整体变暗、轮廓过冲），并用启发式方法进行了修补。修补后的版本是稳定的，但编辑强度过低，实用价值不大。
* 之前的 README 声称“数学上精确的反演”和“已消除水彩效应”，这两点都是错误的，现已删除。

由于没有做定量评估、基线对比和消融实验，无法声称优于现有方法。

## 建议

请使用原始方法：

> V. Kulikov, M. Kleiner, I. Huberman-Spiegelglas, T. Michaeli. **FlowEdit: Inversion-Free Text-Based Editing Using Pre-Trained Flow Models.** ICCV 2025. [arXiv:2412.08629](https://arxiv.org/abs/2412.08629)

```bibtex
@article{kulikov2024flowedit,
  title   = {FlowEdit: Inversion-Free Text-Based Editing Using Pre-Trained Flow Models},
  author  = {Kulikov, Vladimir and Kleiner, Matan and Huberman-Spiegelglas, Inbar and Michaeli, Tomer},
  journal = {arXiv preprint arXiv:2412.08629},
  year    = {2024}
}
```

仅阅读了摘要的相关工作：RF-Inversion（[arXiv:2410.10792](https://arxiv.org/abs/2410.10792)）、DirectEdit（[arXiv:2605.02417](https://arxiv.org/abs/2605.02417)）、Adaptive Projected Guidance（[arXiv:2410.02416](https://arxiv.org/abs/2410.02416)）。

## 许可证

MIT，见 [LICENSE](LICENSE)。代码仅供参考，不再维护。
