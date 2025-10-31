# 🧠 Multi-Method Analysis for Optimal Skin Lesion Detection  
### Combining Images, Metadata, and Fusion Models for Skin Cancer Classification  

This project explores **multi-modal learning approaches** for classifying skin lesions as **benign or malignant** using both image and metadata features. The work investigates three distinct modeling strategies:  

1. **Image-Only Model (ResNet-18):**  
   Utilizes transfer learning from ImageNet to extract visual features from lesion images for binary classification.  

2. **Metadata-Only Model (MLP):**  
   Employs a Multi-Layer Perceptron trained on clinical metadata (age, lesion attributes, etc.) with extensive preprocessing, feature encoding, and selection.  

3. **Fusion-Based Models:**  
   - **Early Fusion:** Combines image embeddings and metadata features at the input level.  
   - **Late Fusion:** Processes image and metadata separately (using ResNet-18 and MLP) before merging their embeddings for final classification.  

---

### 📊 Dataset  
The project uses the **SLICE-3D Dataset** published by the [International Skin Imaging Collaboration (ISIC)](https://www.nature.com/articles/s41597-024-03743-w), which contains:  
- High-quality dermoscopic images collected from 9 dermatology centers worldwide  
- Detailed clinical metadata for each lesion (age, sex, color irregularity, anatomical site, etc.)  
- Diagnostic labels verified through histopathology  

---

### ⚙️ Methodology Overview  
- **Image Preprocessing:** Resizing, normalization, and data augmentation (flips, crops).  
- **Metadata Preprocessing:** One-hot encoding for categorical and standardization for numerical features.  
- **Fusion Techniques:** Both early and late fusion architectures implemented using **PyTorch**.  
- **Optimization:** Binary Cross-Entropy Loss with class balancing, Adam/SGD optimizers, and early stopping.  

---

### 📈 Results Summary  

| Model | Accuracy | Precision | Recall | F1-Score |
|--------|-----------|------------|---------|----------|
| MLP (Metadata Only) | 0.8592 | 0.8824 | 0.8333 | 0.8571 |
| ResNet-18 (Images Only) | 0.6962 | 0.7087 | 0.6962 | 0.6907 |
| MMF – Early Fusion | 0.7468 | 0.9130 | 0.5385 | 0.6774 |
| **MMF – Late Fusion** | **0.8608** | **0.8333** | **0.8974** | **0.8642** |

🩺 **Insight:** The **Late Fusion** model outperformed all others, showing that combining independent feature representations leads to better generalization. Interestingly, metadata alone proved highly predictive, emphasizing its diagnostic value.  

---

### 🧩 Repository Contents  
- `Notebooks/` — Jupyter notebooks for data preprocessing, model training, and evaluation.  
- `Technical Report.pdf` — Full technical report with detailed explanations, architectures, and analysis.    

---

### 💡 Key Takeaways  
- Metadata can carry significant predictive power in medical imaging tasks.  
- Late fusion provides a balanced trade-off between precision and recall.  
- Fusion architectures can enhance model robustness by leveraging complementary modalities.  

---

### 🧰 Tech Stack  
- **Frameworks:** PyTorch, NumPy, Pandas, Scikit-learn  
- **Models:** ResNet-18, MLP  
- **Tools:** Matplotlib, Seaborn, Jupyter Notebook  

---

### 📎 Reference  
This project was developed as part of the group project in [Deep Learning coursework](https://kursuskatalog.au.dk/en/course/134972/Deep-Learning) in the MSc. Computer Engineering program at **Aarhus University**.  

---

### 👩‍💻 Author  
**Lucie Van Roy**
🔗 [GitHub](https://github.com/lucie-vr)

**Ginevra Bozza**
🔗 [GitHub](https://github.com/ginevra-bozza)

**Grazia Cossu**
🔗 [GitHub](https://github.com/Grazia20)

**Saijal Singhal**
🔗 [GitHub](https://github.com/saij19)

---



