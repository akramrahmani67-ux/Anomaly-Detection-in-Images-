🧠 Anomaly Detection in Images
Enhancing Image Anomaly Detection: A Hybrid Framework with Pre-Trained Models and Sequential Neural Networks
<p align="justify"> Image anomaly detection plays a crucial role in visual surveillance and industrial inspection. However, traditional approaches often struggle to detect subtle or context-dependent anomalies. This research introduces a <b>hybrid framework</b> that integrates <b>pre-trained vision models</b> with <b>sequential neural networks</b> (LSTM and GRU), and employs <b>Grad-CAM++</b> for interpretability. </p> <p align="justify"> The proposed method effectively captures both spatial and temporal dependencies, providing enhanced detection accuracy and explainability. Our experiments demonstrate superior performance, reaching <b>AUC values up to 100%</b> on several benchmark datasets. </p>
📊 Benchmark Datasets

The following public datasets were used in this study:

Dataset	Description	Official Link
UCSD Ped1 & Ped2	Pedestrian walkways under various conditions for anomaly motion detection.	UCSD Dataset

Avenue	Street surveillance scenes containing subtle and contextual anomalies.	Avenue Dataset

UMN	Surveillance videos simulating panic scenarios in different environments.	UMN Dataset

Place the datasets inside the /datasets directory before running the notebooks.

⚙️ Feature Extractors

The following pre-trained architectures were used for feature extraction. Each model contributes unique spatial representations for anomaly detection.

Model	Anomaly Features (.npy)	Normal Features (.npy)
CoAtNet-0-RW-224	anomaly_features_coatnet_0_rw_224.npy	normal_features_coatnet_0_rw_224.npy
ConvNeXt-tiny	anomaly_features_convnext_tiny.npy	normal_features_convnext_tiny.npy
MaxViT-tiny-TF-224	anomaly_features_maxvit_tiny_tf_224.npy	normal_features_maxvit_tiny_tf_224.npy
MobileOne-s0	anomaly_features_mobileone_s0.npy	normal_features_mobileone_s0.npy
PoolFormer-s12	anomaly_features_poolformer_s12.npy	normal_features_poolformer_s12.npy
RepVGG-a0	anomaly_features_repvgg_a0.npy	normal_features_repvgg_a0.npy
📁 Pre-Extracted Feature Files

To simplify the workflow, all pre-extracted .npy files are publicly shared.
Each link contains 12 .npy files, corresponding to both normal and abnormal samples extracted using the six feature extractor models.

🔹 Avenue Dataset

📂 Dropbox Link – Avenue

🔹 UCSD Ped1 Dataset

📂 Dropbox Link – UCSD Ped1

🔹 UCSD Ped2 Dataset

📂 Dropbox Link – UCSD Ped2

🔹 UMN Dataset

📂 Dropbox Link – UMN

Each link allows you to download ready-to-use feature sets, eliminating the need to repeat computationally expensive extraction steps.

📈 Sequential Model Results (XLSX)

The folder Sequential_Model_Results_XLSX includes Excel files that summarize the evaluation results of sequential neural networks (LSTM, GRU, and RNN) across all datasets.

Each file presents model performance using various feature extractors and reports metrics such as Accuracy, Precision, Recall, F1-score, and AUC.

Included files:

📘 Results of Sequential Neural Networks for Avenue.xlsx

📘 Results of Sequential Neural Networks for UCSD Ped1.xlsx

📘 Results of Sequential Neural Networks for UCSD Ped2.xlsx

📘 Results of Sequential Neural Networks for UMN.xlsx

These files collectively provide a comparative analysis of different sequential architectures on diverse datasets, demonstrating the robustness and adaptability of the proposed hybrid approach.

🧩 How to Run

This project was implemented and tested in Google Colab using an NVIDIA A100 GPU.

Steps to Reproduce:

Open the provided .ipynb notebooks in Google Colab.

Enable GPU acceleration:
Runtime → Change runtime type → Hardware accelerator → GPU

Execute all cells sequentially to perform:

Feature loading

Sequential model training (LSTM / GRU)

Performance evaluation

Grad-CAM++ visualization

👩‍💻 Authors

Akram Rahmani

Seyfollah Soleimani* (Corresponding Author)

Affiliation:
Department of Computer Engineering, Faculty of Engineering,
Arak University, Arak 38156-8-8349, Iran

📧 Email: s-soleimani@araku.ac.ir

📜 Manuscript Status

This work, titled “Enhancing Image Anomaly Detection: A Hybrid Framework with Pre-Trained Models and Sequential Neural Networks,”
is currently under review at The Visual Computer (Springer Nature).
The repository will be updated following acceptance to include final revisions and full implementation details.

📚 Citation
@article{rahmani2025anomaly,
  title={Enhancing Image Anomaly Detection: A Hybrid Framework with Pre-Trained Models and Sequential Neural Networks},
  author={Rahmani, Akram and Soleimani, Seyfollah},
  journal={The Visual Computer},
  year={2025},
  note={Manuscript under review; repository will be updated upon acceptance. DOI: 10.5281/zenodo.17362656}
}
