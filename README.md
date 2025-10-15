# Anomaly Detection in Images
# Enhancing Image Anomaly Detection: A Hybrid Framework with Pre-Trained Models and Sequential Neural Networks
 <p>Anomaly detection in images is crucial for visual surveillance and industrial inspection. Traditional methods often struggle with subtle or context-aware anomalies.</p>

<p>This paper introduces a hybrid framework combining:</p>
<ul>
  <li>Pre-trained vision models</li>
  <li>Sequence networks (LSTM and GRU)</li>
  <li>The explainable Grad-CAM++ mechanism</li>
</ul>

<p>Our approach significantly outperforms existing literature-based models, achieving an AUC of up to 100% on certain datasets, while providing accurate localization of anomalous regions. The integration of spatial-temporal modeling and gradient-based explanation enhances both accuracy and transparency.</p>

<p>Experimental results on four benchmark datasets (<strong>UCSD Ped1, Ped2, Avenue, and UMN</strong>) demonstrate the superiority of our method, confirming its effectiveness in capturing complex spatial-temporal dependencies.</p>



---

## 📦 Datasets

The following benchmark datasets are used in this work:

- **UCSD Ped1 & Ped2:** [http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm](http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm)  
- **Avenue:** [https://cs-people.bu.edu/hamed/avenue_dataset.html](https://cs-people.bu.edu/hamed/avenue_dataset.html)  
- **UMN:** [https://mha.cs.umn.edu/Movies/UMN_Anomaly_Detection.html](https://mha.cs.umn.edu/Movies/UMN_Anomaly_Detection.html)  

Place datasets under the `datasets/` folder.

---

## ⚡ Feature Extractors

The following pre-trained models were used for feature extraction:

| Model       | Anomaly Features                  | Normal Features                  |
|------------|----------------------------------|---------------------------------|
| CoatNet    | `anomaly_features_coatnet_0_rw_224.npy` | `normal_features_coatnet_0_rw_224.npy` |
| ConvNeXt   | `anomaly_features_convnext_tiny.npy` | `normal_features_convnext_tiny.npy` |
| MaxViT     | `anomaly_features_maxvit_tiny_tf_224.npy` | `normal_features_maxvit_tiny_tf_224.npy` |
| MobileOne  | `anomaly_features_mobileone_s0.npy` | `normal_features_mobileone_s0.npy` |
| PoolFormer | `anomaly_features_poolformer_s12.npy` | `normal_features_poolformer_s12.npy` |
| RepVGG     | `anomaly_features_repvgg_a0.npy` |`normal_features_repvgg_a0.npy`   |

These models are CNN-based or hybrid CNN-Transformer architectures and are used to extract meaningful features from both normal and anomalous frames.

---

## 🚀 How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt


Enhancing Image Anomaly Detection: A Hybrid Framework with Pre-Trained Models and Sequential Neural Networks

Authors:
1. Akram Rahmani
2. Seyfollah Soleimani*

Affiliation:
Department of Computer Engineering, Faculty of Engineering, Arak University, Arak 38156-8-8349, Iran

Corresponding Author:
Dr. Seyfollah Soleimani
📧 Email: s-soleimani@araku.ac.ir

📄 Manuscript Status:
This work, titled “Enhancing Image Anomaly Detection: A Hybrid Framework with Pre-Trained Models and Sequential Neural Networks,” is currently under review for publication in The Visual Computer (Springer Nature) journal. The repository may be updated as the review process progresses or after acceptance to include final revisions, datasets, and code documentation.


