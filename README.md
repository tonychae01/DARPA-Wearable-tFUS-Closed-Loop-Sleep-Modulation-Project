
# DARPA-funded, Close-loop Wearable Transcranial Ultrasound-based Sleep Modulation System

**Dates of Participation:** March 10, 2024 – Present  
**Laboratory:** Wang Biomedical Research Laboratory, UT Austin


---

## Project Overview
A real-time closed-loop sleep modulation platform integrating multi-modal EEG/EMG/EOG recordings with focused ultrasound stimulation. Developed under Dr. Huiliang Wang’s DARPA-funded research to stabilize sleep stages via AI-driven classification and a bio-adhesive neural interface.

---

## Key Contributions
- **Advanced Feature Engineering:** Designed and implemented a feature augmentation pipeline extracting 220 robust features (statistical moments, spectral power, entropy, complexity metrics). This reduced computational overhead by 85% while preserving classification performance.
- **Comparative Modeling:** Benchmarked traditional classifiers (Random Forest, SVM, XGBoost, ANN) against deep architectures (Parallel 1D CNN+LSTM, Transformers, Liquid Neural Network). Achieved up to 92.3% Group K - cross-validated accuracy with the CNN+LSTM model.
- **Transfer Learning for Generalization:** Pretrained on the ANPHY‑29 open-source EEG dataset and fine-tuned on NEUSLeeP data, yielding 89.6% accuracy on unseen subjects and mitigating overfitting.
- **Naïve Bayesian Integration (NBI) Smoothing:** Introduced a Bayesian smoothing layer inspired by prosthetic control to reduce transient misclassifications, improving stability in real-time applications by 45%.
- **Real-Time GUI & Closed‑Loop Control:** Developed a PyQt5 interface with live EEG/EOG/EMG visualization, hypnogram plotting, and automated ultrasound triggers via serial communication.


![image](https://github.com/user-attachments/assets/2efd78e8-55fa-48bd-b009-c983bf789852)
![image](https://github.com/user-attachments/assets/604750d1-5221-405e-aa95-3d4b76c30091)

---

## Methodology
1. **Data Acquisition**  
   - Six‑channel NEUSLeeP bio‑adhesive interface (4 EEG, 1 EMG, 1 EOG) and BrainVision 32‑channel system.  
   - IRB‑approved PTSD dataset from military personnel, annotated per AASM guidelines.

2. **Preprocessing**  
   - Bandpass filtering (0.5–45 Hz) and notch filtering at 60 Hz.  
   - Artifact rejection via ICA and threshold‑based outlier detection.  
   - Sliding 30 s windows at 512 Hz sampling rate.
![image](https://github.com/user-attachments/assets/527fcae3-e189-4a10-a225-fa132037d03e)

3. **Feature Engineering**  
   - **Time-domain:** mean, variance, skewness, kurtosis, entropy.  
   - **Frequency-domain:** power spectral density (4–30 Hz bands), peak frequency, band ratios (α/θ, γ/δ).  
   - **Complexity metrics:** fractal dimension, Hjorth parameters.  
   - Combined current and previous‑window features into a 220‑dimensional vector.

4. **Model Training & Evaluation**  
   - **Traditional ML:** Random Forest (100 trees, Gini), SVM (RBF, C=1.0, γ='scale'), XGBoost (η=0.05, max_depth=8).  
   - **Deep Learning:** Parallel 1D CNN (filters=64, kernel_size=3) + LSTM (units=128) with dropout=0.3.  
   - 29‑fold cross‑validation; metrics: accuracy, F1‑score, Cohen’s κ.
   
![image](https://github.com/user-attachments/assets/f853d026-6cba-4508-9129-d88af1505e68)

5. **Transfer Learning**  
   - Pretrained CNN backbone on ANPHY‑29; fine‑tuned the final layers on NEUSLeeP with learning rate=1e‑4.

6. **NBI Smoothing**  
   - Bayesian integration over a 3‑epoch sliding window to stabilize predictions and reduce oscillatory errors.
  
     
![image](https://github.com/user-attachments/assets/463531ce-6f0d-4837-b61a-303985141241)
![image](https://github.com/user-attachments/assets/73fc040f-0cf6-400c-bd77-f9c5126c8a99)
![image](https://github.com/user-attachments/assets/41146afb-ef1f-4180-aec1-7648bebdb708)

7. **Real‑Time System & GUI**  
   - PyQt5 application with PyQtGraph for live signal plotting and hypnogram rendering.  
   - Automated stimulation logic: N2 spindle detection (YASA) + thresholded sleep probability → ultrasound trigger via serial port.
   - 
![image](https://github.com/user-attachments/assets/d4967e3a-f8a8-4628-a3a4-8ff945490b84)

![image](https://github.com/user-attachments/assets/5c803071-9271-4121-bb9f-54670ac703c5)
![image](https://github.com/user-attachments/assets/72addf19-bfc0-47c2-867e-0ca6daeeeb7c)

![image](https://github.com/user-attachments/assets/678cf1c0-3576-4fb5-90ea-9882de4f25f7)


---

## Results
- **Computational Efficiency:** Reduced end‑to‑end training time from 116 hours to 54 minutes with optimized features.  
- **Classification Performance:** 92.3% accuracy (CNN+LSTM), 91.1% (XGBoost), 89.6% (transfer‑learned CNN).  
- **Prediction Stability:** 45% reduction in misclassification rate with NBI smoothing.  
- **Generalization:** 89.6% on held‑out subjects through transfer learning.

---

## Technologies & Tools
- **Frameworks:** Python, PyTorch, Scikit‑learn, XGBoost, PyQt5, PyQtGraph, YASA, MNE.  
- **Hardware:** NEUSLeeP bio‑adhesive elastomer electrodes, Focused Ultrasound Stimulation device.  
- **Protocols:** IRB‑approved data collection, AASM sleep staging.

---

## Usage
1. **Clone the repository:**  
   ```bash
   git clone https://github.com/yourusername/sleep-modulation-system.git
   ```
2. **Install dependencies:**  
   ```bash
   pip install -r requirements.txt
   ```
3. **Launch the GUI:**  
   ```bash
   python realtimecode_final.py
   ```
4. **Configuration:** Edit `config.json` to set serial port, probability thresholds, and stimulation parameters.

---

## Acknowledgments
DARPA, Dr. Huiliang Wang, UT Austin Biomedical Engineering Department, NEUSLeeP research team.

