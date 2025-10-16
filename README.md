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

Anomaly Detection Feature Extractors – Public Datasets

For ease of feature extraction, you can directly access the pre-extracted .npy feature files for each dataset from the links below. These files contain both normal and abnormal feature samples generated using various pre-trained models such as ConvNeXt-tiny, RepVGG-a0, MobileOne-s0, PoolFormer-s12, MaxViT-tiny-TF-224, and CoAtNet-0-RW-224.

Avenue Dataset
 🔗 https://www.dropbox.com/scl/fo/uvj2i4kaqnj425rb232vt/ADTsWLQjx-Fi-eJ9C7c9cVg?rlkey=02g9u1c5ejkhi2umk0pbs96fc&st=be1ubtml&dl=0
UCSD Ped1 Dataset
🔗 https://www.dropbox.com/scl/fo/j5zya2mz4xfqyqqgv7wac/AFxVNsxkhfVwjyf3jQuEQuM?rlkey=1ge2yikbbpw8a7smwbq63p3if&st=172oj2nx&dl=0
UCSD Ped2 Dataset
🔗 https://www.dropbox.com/scl/fo/lbse8zt94o24i8fyvnkd2/AMA66KNLLDbxFt3ij4EuLco?rlkey=09m8m0tygstpzow3b0n5uj5jk&st=oe274j3o&dl=0)
UMN Dataset
🔗 https://www.dropbox.com/scl/fo/095z706yk3rzhg6wd0yxa/ABbDGuAP4xsA9AMyyrRP8s0?rlkey=1ulqi6z9vzoqjxdnyqkqdazi2&st=9ieth52p&dl=0)

---

 <h2>Enhancing Image Anomaly Detection: A Hybrid Framework with Pre-Trained Models and Sequential Neural Networks</h2>

<h3>How to Run</h3>

<p>
This project was developed and executed in the <b>Google Colab</b> environment using an <b>NVIDIA A100 GPU</b>. 
All experiments, including feature extraction, sequential modeling (LSTM and GRU), and visualization with Grad-CAM++, 
were implemented and tested directly in Colab without requiring any manual hardware configuration.
</p>

<p>
To reproduce the results:
</p>

<ol>
  <li>Open the provided <code>.ipynb</code> files in Google Colab.</li>
  <li>Ensure GPU acceleration is enabled (<b>Runtime → Change runtime type → Hardware accelerator → GPU</b>).</li>
  <li>Execute the notebook cells in sequence to reproduce training, evaluation, and visualization steps.</li>
</ol>

<h3>Authors</h3>
<p>
1. <b>Akram Rahmani</b><br>
2. <b>Seyfollah Soleimani*</b>
</p>

<h3>Affiliation</h3>
<p>
Department of Computer Engineering, Faculty of Engineering,<br>
Arak University, Arak 38156-8-8349, Iran
</p>

<h3>Corresponding Author</h3>
<p>
<b>Dr. Seyfollah Soleimani</b><br>
📧 Email: <a href="mailto:s-soleimani@araku.ac.ir">s-soleimani@araku.ac.ir</a>
</p>

<h3>Manuscript Status</h3>
<p>
This work, titled <i>“Enhancing Image Anomaly Detection: A Hybrid Framework with Pre-Trained Models and Sequential Neural Networks,”</i> 
is currently under review for publication in <b>The Visual Computer (Springer Nature)</b> journal. 
The repository may be updated as the review process progresses or after acceptance to include 
final revisions and detailed code documentation.
</p>

<h3>Citation</h3>
<pre>
@article{rahmani2025anomaly,
  title={Enhancing Image Anomaly Detection: A Hybrid Framework with Pre-Trained Models and Sequential Neural Networks},
  author={Rahmani, Akram and Soleimani, Seyfollah},
  journal={The Visual Computer},
  year={2025},
  note={Manuscript under review; repository may be updated upon acceptance. DOI: 10.5281/zenodo.17362656}
}
</pre>

