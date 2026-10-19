# Flow Inversion Img2Img (ComfyUI) — archived

**English** | [Русский](README_ru.md) | [中文](README_cn.md)

> **Status: closed. The research goal was not achieved.**
> If you need text-based editing with flow-matching models, use **[FlowEdit](https://arxiv.org/abs/2412.08629)** instead of this repository.

## What this project was

The goal was a full research project: a mathematically corrected inversion of the flow-matching ODE (Anima, Wan2.1, Flux, SD3) that allows mask-free Img2Img editing in ComfyUI while keeping fine details (line art, fingers, cloth folds, background).

## What it turned out to be

The code records the inversion trajectory and replays it step by step (`x ← x − d_k`). Reconstruction is therefore exact **by construction**, not because the ODE was shown to be invertible. On top of the replay, a FlowEdit-like delta `v_target − v_source` is added. This is structurally the same idea as FlowEdit, so the project has no real novelty over it.

Over four days of experiments (one model, one image, one seed) the following was found:

* Small edits work: hair color and eye color changed while the structure stayed intact.
* Large regions (for example a white dress to black) cannot be recolored. The structure of the source latent holds the color, and a stronger edit gives destruction instead of the target color.
* A strong edit produces a "watercolor" look: a glowing halo and washed-out forms. It is an artifact (CFG oversaturation / off-manifold collapse), uncontrollable and not usable for any practical task.
* Several bugs and pathologies were found and patched with heuristics (wrong σ index, CFG mismatch between branches, global darkening, edge overshoot). The patched version is stable, but its edit strength is too low to be useful.
* The previous README claimed a "mathematically exact inversion" and a "fixed watercolor effect". Both claims were wrong and have been removed.

No quantitative evaluation, baselines or ablations were done, so no claim of improvement over existing methods can be made.

## Recommendation

Use the original method:

> V. Kulikov, M. Kleiner, I. Huberman-Spiegelglas, T. Michaeli. **FlowEdit: Inversion-Free Text-Based Editing Using Pre-Trained Flow Models.** ICCV 2025. [arXiv:2412.08629](https://arxiv.org/abs/2412.08629)

```bibtex
@article{kulikov2024flowedit,
  title   = {FlowEdit: Inversion-Free Text-Based Editing Using Pre-Trained Flow Models},
  author  = {Kulikov, Vladimir and Kleiner, Matan and Huberman-Spiegelglas, Inbar and Michaeli, Tomer},
  journal = {arXiv preprint arXiv:2412.08629},
  year    = {2024}
}
```

Related work that was read only at abstract level: RF-Inversion ([arXiv:2410.10792](https://arxiv.org/abs/2410.10792)), DirectEdit ([arXiv:2605.02417](https://arxiv.org/abs/2605.02417)), Adaptive Projected Guidance ([arXiv:2410.02416](https://arxiv.org/abs/2410.02416)).

## License

MIT, see [LICENSE](LICENSE). The code is kept for reference only and is not maintained.
