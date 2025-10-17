 Anomaly Detection in Images  
## Enhancing Image Anomaly Detection: A Hybrid Framework with Pre-Trained Models and Sequential Neural Networks

### Abstract
Image anomaly detection is essential for visual surveillance and industrial inspection. Traditional approaches often fail on subtle or context-dependent anomalies.  
This work proposes a **hybrid framework** that combines pre-trained vision backbones for spatial feature extraction with sequence models (LSTM, GRU, Simple RNN) to capture temporal dynamics, and uses Grad-CAM++ for interpretable localization of anomalous regions.  
Experimental results on multiple benchmarks demonstrate strong performance and reliable localization of anomalies.

---

## 📊 Benchmark Datasets
The experiments use four publicly available datasets. Place original dataset files under the `/datasets` directory to re-run the full pipeline (feature extraction + sequential modeling).

- **UCSD Ped1 & Ped2** — Pedestrian walkways under different conditions for motion anomaly detection.  
  [Official page](http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm)

- **Avenue** — Street surveillance scenes containing subtle and contextual anomalies.  
  [Official page](https://cs-people.bu.edu/hamed/avenue_dataset.html)

- **UMN** — Simulated panic scenarios recorded in several environments.  
  [Official page](https://mha.cs.umn.edu/Movies/UMN_Anomaly_Detection.html)

---

## ⚙️ Feature Extractors
Several state-of-the-art pre-trained backbones were used to extract spatial features from frames, which were then provided to sequence models for temporal modeling.

| Model | Example anomaly features file (.npy) | Example normal features file (.npy) |
|-------|--------------------------------------|-------------------------------------|
| CoAtNet-0-RW-224 | anomaly_features_coatnet_0_rw_224.npy | normal_features_coatnet_0_rw_224.npy |
| ConvNeXt-tiny | anomaly_features_convnext_tiny.npy | normal_features_convnext_tiny.npy |
| MaxViT-tiny-TF-224 | anomaly_features_maxvit_tiny_tf_224.npy | normal_features_maxvit_tiny_tf_224.npy |
| MobileOne-s0 | anomaly_features_mobileone_s0.npy | normal_features_mobileone_s0.npy |
| PoolFormer-s12 | anomaly_features_poolformer_s12.npy | normal_features_poolformer_s12.npy |
| RepVGG-a0 | anomaly_features_repvgg_a0.npy | normal_features_repvgg_a0.npy |

These models are CNN- or CNN–Transformer hybrid architectures selected for robust spatial representations.

---

## 📁 Pre-Extracted Feature Files (Download)
Pre-extracted `.npy` feature files are shared via Dropbox. Each link contains **12 .npy files** representing normal and abnormal samples for the corresponding dataset.

- **Avenue**: [Download link](https://www.dropbox.com/scl/fo/uvj2i4kaqnj425rb232vt/ADTsWLQjx-Fi-eJ9C7c9cVg?rlkey=02g9u1c5ejkhi2umk0pbs96fc&st=be1ubtml&dl=0)  
- **UCSD Ped1**: [Download link](https://www.dropbox.com/scl/fo/j5zya2mz4xfqyqqgv7wac/AFxVNsxkhfVwjyf3jQuEQuM?rlkey=1ge2yikbbpw8a7smwbq63p3if&st=172oj2nx&dl=0)  
- **UCSD Ped2**: [Download link](https://www.dropbox.com/scl/fo/lbse8zt94o24i8fyvnkd2/AMA66KNLLDbxFt3ij4EuLco?rlkey=09m8m0tygstpzow3b0n5uj5jk&st=oe274j3o&dl=0)  
- **UMN**: [Download link](https://www.dropbox.com/scl/fo/095z706yk3rzhg6wd0yxa/ABbDGuAP4xsA9AMyyrRP8s0?rlkey=1ulqi6z9vzoqjxdnyqkqdazi2&st=9ieth52p&dl=0)  

> Each link contains `.npy` files and a `.txt` describing the mapping of feature extractors to normal/abnormal samples.

---

## 📊 Sequential Model Results (XLSX)
The `Sequential_Model_Results_XLSX` folder includes Excel files summarizing the evaluation of sequential models on each dataset.

- Results of Sequential Neural Networks for Avenue.xlsx  
- Results of Sequential Neural Networks for UCSD Ped1.xlsx  
- Results of Sequential Neural Networks for UCSD Ped2.xlsx  
- Results of Sequential Neural Networks for UMN.xlsx  

Metrics reported:

- Accuracy  
- Precision  
- Recall  
- F1-Score  
- ROC-AUC

  # Best results from pre-trained models

The folder `Best results from pre-trained models` contains the output figures for anomaly detection:

- **GRU_Anomaly_Avenue.png**
- **UCSD_Ped1_LSTM+GRU_Anomaly.png**
- **UCSD_Ped2_GRU_Anomaly.png**
- **UMN_LSTM_Anomaly.png**

These images illustrate the best results obtained from the pre-trained feature extractors and sequential models.
   ## 👩‍💻 Authors  

**Author:** Akram Rahmani  
**Email:** akram.rahmani67@gmail.com  

**Corresponding Author:** Seyfollah Soleimani  
**Email:** s-soleimani@araku.ac.ir  

**Affiliation:** Department of Computer Engineering, Faculty of Engineering, Arak University, Arak 38156-8-8349, Iran  

---

## 📜 Manuscript Status
**Title:** Enhancing Image Anomaly Detection: A Hybrid Framework with Pre-Trained Models and Sequential Neural Networks  
**Current status:** Under review at *The Visual Computer* (Springer Nature)  
> The repository will be updated following acceptance to include final revisions and full code release.

---

## 📚 Citation
```bibtex
@article{rahmani2025anomaly,
  title={Enhancing Image Anomaly Detection: A Hybrid Framework with Pre-Trained Models and Sequential Neural Networks},
  author={Rahmani, Akram and Soleimani, Seyfollah},
  journal={The Visual Computer},
  year={2025},
  note={Manuscript under review; repository will be updated upon acceptance. DOI: 10.5281/zenodo.17362656}
}

