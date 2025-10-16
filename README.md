# 🧠 Anomaly Detection in Images  
## Enhancing Image Anomaly Detection: A Hybrid Framework with Pre-Trained Models and Sequential Neural Networks  

### **Abstract**  
Image anomaly detection is a crucial requirement for visual surveillance and industrial quality inspection. However, most prior works fail to perform well in cases of subtle or contextual anomalies.  
In this paper, a **hybrid framework** is introduced that incorporates **pre-trained vision models**, **sequence-based networks (LSTM, GRU)**, and the **explainable mechanism Grad-CAM++**.  
The pre-trained models learn rich spatial representations from image frames, while sequence-based models learn temporal dependencies among image frames. Grad-CAM++ provides interpretability by localizing regions contributing to anomaly decisions.  

---

### **1. Introduction**  
Anomaly detection in images aims to identify patterns or regions that deviate from normal behavior. It is widely used in domains such as:  
- Industrial defect detection  
- Surveillance video analysis  
- Medical imaging  

Traditional methods often rely on handcrafted features, which are limited in capturing complex spatial and temporal dependencies. The proposed framework leverages the strengths of **deep pre-trained architectures** and **sequence-based temporal learning**, resulting in robust and explainable anomaly detection.

---

### **2. Datasets Used**
The framework was evaluated on several benchmark datasets:  

| Dataset | Description | Link |
|----------|--------------|------|
| UCSD Ped1 | Pedestrian dataset for anomaly detection | [UCSD Ped1 Dataset](https://www.svcl.ucsd.edu/projects/anomaly/dataset.htm) |
| UCSD Ped2 | Similar to Ped1 but with different camera angles | [UCSD Ped2 Dataset](https://www.svcl.ucsd.edu/projects/anomaly/dataset.htm) |
| Avenue | Video dataset for abnormal events | [CUHK Avenue Dataset](https://github.com/StevenLiuWen/ano_pred_cvpr2018) |
| ShanghaiTech | Large-scale dataset for anomaly detection | [ShanghaiTech Dataset](https://svip-lab.github.io/dataset/campus_dataset.html) |

Each dataset provides both training and testing sequences, allowing separate evaluation on normal and abnormal behaviors.

---

### **3. Proposed Hybrid Framework**
The proposed system consists of three main stages:

1. **Feature Extraction**  
   - Pre-trained CNNs such as **VGG16**, **ResNet50**, and **EfficientNet** are used to extract deep spatial features from frames.  
   - These features represent texture, structure, and contextual information of images.

2. **Temporal Modeling**  
   - Sequential neural networks (**LSTM** and **GRU**) capture the **temporal evolution** of visual patterns across consecutive frames.  
   - This helps detect motion-based or context-based anomalies.

3. **Explainability Module**  
   - **Grad-CAM++** visualizes the key regions responsible for anomaly detection decisions, enhancing interpretability and trustworthiness of the model.

---

### **4. Implementation Details**

- **Programming Environment:** Python (TensorFlow / Keras)  
- **Hardware:** NVIDIA GPU (A100 used for experiments)  
- **Optimizer:** Adam  
- **Loss Function:** Binary Cross-Entropy  
- **Metrics:** Accuracy, Precision, Recall, F1-score, and AUC  

Data augmentation and normalization were applied to improve generalization.

---

### **5. Experimental Results**

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-score (%) | AUC |
|--------|---------------|----------------|--------------|---------------|-----|
| VGG16 + LSTM | 95.6 | 95.2 | 94.8 | 95.0 | 0.97 |
| ResNet50 + GRU | 96.1 | 95.8 | 95.4 | 95.6 | 0.98 |
| EfficientNet + LSTM | **97.3** | **96.8** | **96.9** | **96.8** | **0.99** |

The hybrid design achieved significant improvements over single-model baselines, especially in detecting subtle contextual anomalies.

---

### **6. Visualization and Explainability**
Grad-CAM++ heatmaps were generated for both normal and abnormal samples.  
- **Normal Frames:** Showed uniformly distributed activations.  
- **Anomalous Frames:** Highlighted localized regions corresponding to unusual objects or motions (e.g., vehicles in pedestrian zones, running individuals, or abnormal interactions).

---

### **7. Conclusion**
This work presents a **hybrid and interpretable deep learning framework** for anomaly detection in images and videos.  
By combining **spatial feature extraction (CNN)**, **temporal modeling (LSTM/GRU)**, and **explainability (Grad-CAM++)**, the system achieves high detection accuracy and visual interpretability.  
Future work may focus on real-time optimization and unsupervised anomaly learning.

---

### **8. Citation**
If you use this framework in your research, please cite as:  

```
@article{rahmani2025anomaly,
  title={Enhancing Image Anomaly Detection: A Hybrid Framework with Pre-Trained Models and Sequential Neural Networks},
  author={Rahmani, Akram and Soleimani, [Co-author name]},
  journal={Signal Processing and Renewable Energy},
  year={2025}
}
```

---

### **9. Contact**
For further questions or collaborations:  
📧 **akramrahmani.research@gmail.com**
