Anomaly Detection in Images
Enhancing Image Anomaly Detection: A Hybrid Framework with Pre-Trained Models and Sequential Neural Networks

Abstract
Image anomaly detection is essential for visual surveillance and industrial inspection. Traditional approaches often fail on subtle or context-dependent anomalies. This work proposes a hybrid framework that combines pre-trained vision backbones for spatial feature extraction with sequence models (LSTM, GRU, Simple RNN) to capture temporal dynamics, and uses Grad-CAM++ for interpretable localization of anomalous regions. Experimental results on multiple benchmarks demonstrate strong performance and reliable localization of anomalies.

📊 Benchmark Datasets

The experiments use four publicly available datasets. Place original dataset files under the /datasets directory if you want to re-run the full pipeline (feature extraction + sequential modeling).

UCSD Ped1 & Ped2 — Pedestrian walkways under different conditions for motion anomaly detection.
Official page: http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm

Avenue — Street surveillance scenes containing subtle and contextual anomalies.
Official page: https://cs-people.bu.edu/hamed/avenue_dataset.html

UMN — Simulated panic scenarios recorded in several environments.
Official page: https://mha.cs.umn.edu/Movies/UMN_Anomaly_Detection.html

⚙️ Feature Extractors

We used several state-of-the-art pre-trained backbones to extract spatial features from frames. These extracted features are then provided to sequence models for temporal modeling.

Model	Example anomaly features file (.npy)	Example normal features file (.npy)
CoAtNet-0-RW-224	anomaly_features_coatnet_0_rw_224.npy	normal_features_coatnet_0_rw_224.npy
ConvNeXt-tiny	anomaly_features_convnext_tiny.npy	normal_features_convnext_tiny.npy
MaxViT-tiny-TF-224	anomaly_features_maxvit_tiny_tf_224.npy	normal_features_maxvit_tiny_tf_224.npy
MobileOne-s0	anomaly_features_mobileone_s0.npy	normal_features_mobileone_s0.npy
PoolFormer-s12	anomaly_features_poolformer_s12.npy	normal_features_poolformer_s12.npy
RepVGG-a0	anomaly_features_repvgg_a0.npy	normal_features_repvgg_a0.npy

These models are CNN- or CNN–Transformer hybrid architectures selected for robust spatial representations.

📁 Pre-Extracted Feature Files (Download)

To make replication and further experiments easier, pre-extracted .npy feature files are shared via Dropbox.
Each Dropbox link contains 12 .npy files that represent feature sets for normal and abnormal samples extracted from the corresponding dataset (features extracted using the six models listed above). You can download the files and directly proceed to sequence modeling without re-running feature extraction.

Avenue

https://www.dropbox.com/scl/fo/uvj2i4kaqnj425rb232vt/ADTsWLQjx-Fi-eJ9C7c9cVg?rlkey=02g9u1c5ejkhi2umk0pbs96fc&st=be1ubtml&dl=0

UCSD Ped1

https://www.dropbox.com/scl/fo/j5zya2mz4xfqyqqgv7wac/AFxVNsxkhfVwjyf3jQuEQuM?rlkey=1ge2yikbbpw8a7smwbq63p3if&st=172oj2nx&dl=0

UCSD Ped2

https://www.dropbox.com/scl/fo/lbse8zt94o24i8fyvnkd2/AMA66KNLLDbxFt3ij4EuLco?rlkey=09m8m0tygstpzow3b0n5uj5jk&st=oe274j3o&dl=0

UMN

https://www.dropbox.com/scl/fo/095z706yk3rzhg6wd0yxa/ABbDGuAP4xsA9AMyyrRP8s0?rlkey=1ulqi6z9vzoqjxdnyqkqdazi2&st=9ieth52p&dl=0

Note: Each of the above links contains the .npy files (12 files per dataset) and an accompanying .txt describing which file corresponds to which extractor and whether it contains normal or abnormal samples.

📊 Sequential Model Results (XLSX)

The Sequential_Model_Results_XLSX folder includes Excel files that summarize the experimental evaluation of sequential models on each dataset. Each workbook reports multiple model configurations and standard metrics.

Included files:

Results of Sequential Neural Networks for Avenue.xlsx

Results of Sequential Neural Networks for UCSD Ped1.xlsx

Results of Sequential Neural Networks for UCSD Ped2.xlsx

Results of Sequential Neural Networks for UMN.xlsx

Metrics reported (per configuration / per feature extractor):

Accuracy

Precision

Recall

F1-Score

ROC-AUC

(Optional) Confusion matrix values, per-class breakdowns, and threshold analysis sheets

🧩 How to Run (Reproducibility)

This project was developed and executed using Google Colab (NVIDIA A100 GPU available in the environment used for experiments). Notebooks are provided to reproduce the main steps.

Steps

Open the provided .ipynb notebooks in Google Colab.

Enable GPU acceleration: Runtime → Change runtime type → Hardware accelerator → GPU.

If you want to use the pre-extracted features, download the .npy files from the Dropbox links above and place them in /content/features/<dataset>/ or update the notebook paths accordingly.

Execute the cells sequentially to:

Load features (.npy)

Prepare sequences (windowing, normalization)

Train sequential models (LSTM / GRU / RNN)

Evaluate performance and export results to Excel (if desired)

Generate Grad-CAM++ visualizations for selected frames

The notebooks include parameters at the top (sequence length, batch size, learning rate) so you can re-run experiments or change hyperparameters.

🔧 Code Organization (suggested)
/
├─ notebooks/
│  ├─ 01_feature_extraction.ipynb          # optional: if you want to re-extract features
│  ├─ 02_sequence_preparation_and_training.ipynb
│  └─ 03_evaluation_and_visualization.ipynb
├─ features/                               # (optional) place downloaded .npy files here
│  ├─ avenue/
│  ├─ ucsd_ped1/
│  ├─ ucsd_ped2/
│  └─ umn/
├─ results/
│  └─ Sequential_Model_Results_XLSX/
│     ├─ Results of Sequential Neural Networks for Avenue.xlsx
│     ├─ Results of Sequential Neural Networks for UCSD Ped1.xlsx
│     ├─ Results of Sequential Neural Networks for UCSD Ped2.xlsx
│     └─ Results of Sequential Neural Networks for UMN.xlsx
├─ README.md
└─ requirements.txt

🧪 Experimental Notes & Reproducibility Tips

Preprocessing: Frames were resized and normalized according to the pre-trained backbones’ expected input.

Feature extraction was performed using each backbone’s pre-trained weights (ImageNet or equivalent).

Sequence length (temporal window), stride, and normalization significantly affect results — those hyperparameters are documented at the top of the notebooks.

When using the shared .npy files, confirm that your notebook’s expected file names match those in the downloaded folder (a .txt file within each Dropbox package lists file-name mappings).

👩‍💻 Authors

Akram Rahmani

Seyfollah Soleimani* (Corresponding Author)

Affiliation:
Department of Computer Engineering, Faculty of Engineering,
Arak University, Arak 38156-8-8349, Iran

Corresponding Author Email: s-soleimani@araku.ac.ir

📜 Manuscript Status

Title: Enhancing Image Anomaly Detection: A Hybrid Framework with Pre-Trained Models and Sequential Neural Networks
Current status: Under review at The Visual Computer (Springer Nature). The repository will be updated following acceptance to include final revisions and full code release.

📚 Citation
@article{rahmani2025anomaly,
  title={Enhancing Image Anomaly Detection: A Hybrid Framework with Pre-Trained Models and Sequential Neural Networks},
  author={Rahmani, Akram and Soleimani, Seyfollah},
  journal={The Visual Computer},
  year={2025},
  note={Manuscript under review; repository will be updated upon acceptance. DOI: 10.5281/zenodo.17362656}
}
