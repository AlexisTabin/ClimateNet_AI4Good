# AI4Good ClimateNet: Final Report

**ETH Zurich - AI4Good 2022**

**Authors:**
- Ying Xue (yinxue@ethz.ch)
- Emilia Arens (earens@ethz.ch)
- Katarzyna Krasnopolska (krasnopk@ethz.ch)
- Alexis Tabin (atabin@ethz.ch)
- Amalie Kjaer (akjaer@ethz.ch)


## Abstract

This project builds on Prabhat et al. [7] to predict the occurrence of extreme weather events — tropical cyclones (TC) and atmospheric rivers (AR) — using expert-labeled world maps with 16 types of atmospheric data. Main challenges include severe class imbalance. We assess multiple neural network architectures and explore Curriculum Learning (CL) for improving performance. CGNet and U-net provide the best results, and CL significantly improves TC detection.



## 1. Introduction

Understanding how climate change affects extreme weather events like TCs and ARs is crucial. This project aims to leverage deep learning to learn patterns of TCs and ARs, overcoming heuristic limitations and class imbalance. 

- **Dataset**: Expert-labeled maps (source: [ClimateNet Dataset](https://portal.nersc.gov/project/ClimateNet/))
- **Challenge**: High class imbalance (93.7% BG, 5.8% AR, 0.5% TC)



## 2. Related Work

- **Heuristics-based models**: High uncertainty, dataset-specific tuning.
- **Deep Learning (DL)**: Baseline model — DeepLabv3++ on selected features (TMQ, U850, V850, PRECT).



## 3. Method

### 3.1 Extending the Baseline

#### 3.1.1 Feature Selection

- **Groups Evaluated**:
  - **Group I** (Baseline): TMQ, U850, V850, PRECT.
  - **Group II** (ANOVA): TMQ, U850, UBOT, VBOT.
  - **Group III** (Mutual Info): QREFHT, PRECT, Z200, ZBOT.

#### 3.1.2 Model Architectures

- **DeepLabv3++**: Baseline.
- **UPerNet**: Failed to detect TCs.
- **U-net**: Good performance; strong for class imbalance.
- **CGNet**: Best trade-off between performance and speed.

#### 3.1.3 Loss Functions

- **Evaluated**: Jaccard, Dice, Weighted CE, CE.
- **Conclusion**: CE achieves best mean IoU but underperforms on TC class.

### 3.2 Tackling Complexity - Curriculum Learning (CL)

- **Stages defined**: From simple (BG only) to complex (BG + AR + TC).
- **Curricula**:
  - **CL I**: Gradual complexity increase.
  - **CL II & III**: More fine-grained, but with trade-offs in AR detection.



## 4. Results

### 4.1 Extending the Baseline

| Feature Group | BG IoU | TC IoU | AR IoU | Mean IoU |
|---------------|--------|--------|--------|----------|
| **Group I**   | 0.941  | 0.342  | 0.401  | 0.564    |
| Group II      | 0.940  | 0.310  | 0.420  | 0.560    |
| Group III     | 0.310  | 0.000  | 0.000  | 0.100    |

| Model         | BG IoU | TC IoU | AR IoU | Mean IoU |
|---------------|--------|--------|--------|----------|
| DeepLabv3++   | 0.938  | 0.283  | 0.397  | 0.542    |
| UPerNet       | 0.936  | 0.000  | 0.396  | 0.443    |
| **U-net**     | 0.941  | **0.359**| **0.404**| **0.568**|
| CGNet         | 0.941  | 0.342  | 0.401  | 0.564    |

| Loss Type     | BG IoU | TC IoU | AR IoU | Mean IoU |
|---------------|--------|--------|--------|----------|
| Jaccard       | 0.942  | **0.348**| 0.403  | 0.564    |
| Dice          | 0.922  | 0.332  | 0.377  | 0.544    |
| Weighted CE   | 0.857  | 0.226  | 0.272  | 0.452    |
| **CE**        | **0.957**| 0.312  | **0.401**| **0.584**|

### 4.2 Curriculum Learning

#### 4.2.1 Standard vs. Patch vs. CL

| Setting                | BG IoU | TC IoU | AR IoU | Mean IoU |
|-----------------------|--------|--------|--------|----------|
| **Baseline**           | 0.9389 | 0.2441 | 0.3910 | 0.5247   |
| **Our Base Model**     | 0.9542 | **0.4856**| 0.3383 | **0.5927**|
| Patch 224x224          | 0.9207 | 0.0000 | 0.5514 | 0.4907   |
| **CL I**               | 0.9324 | 0.3162 | 0.3572 | **0.5353**|

#### 4.2.2 CL II & III

| Curriculum | BG IoU | TC IoU | AR IoU | Mean IoU |
|------------|--------|--------|--------|----------|
| CL II      | 0.934  | 0.3501 | 0.1381 | 0.4740   |
| CL III     | **0.941**| 0.3483 | 0.0855 | 0.4583   |



## 5. Discussion

- **Curriculum Learning** helps improve TC detection.
- **Trade-offs**: Better TC detection but worse AR detection in some curricula.
- **Class imbalance** remains a challenge.
- **Spatial information loss** in patches affects performance.



## 6. Conclusion

- U-net and CGNet outperform the baseline.
- Curriculum Learning improves detection of rare classes.
- Key takeaways:
  - Patch size matters (224 x 224 ideal but limited by data availability).
  - Architecture choice impacts performance (DeepLabv3++ for CL I).
  - Curriculum structure significantly affects outcomes.
- **Open-source code**: [GitHub repository](https://github.com/earens/ClimateNet_AI4Good)



## 7. Future Work

- **Curriculum tuning**: Reducing distribution shifts and model memory issues.
- **New difficulty metrics** beyond class representation.
- **Combining models**: Ensemble methods to address class-specific weaknesses.
- **Spatial encoding** to preserve context in patches.



## References

See the full list of references in the [original paper](https://portal.nersc.gov/project/ClimateNet/).
