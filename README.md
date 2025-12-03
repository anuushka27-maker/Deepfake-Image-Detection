Deepfake Image Detection — Hybrid CNN + ResNet50

An end-to-end Deep Learning system that detects whether a facial image is Real or AI-Generated (Fake).
I built everything from scratch — from collecting and cleaning a dataset of ~141K images, designing and training a Hybrid CNN + ResNet50 architecture, performing detailed evaluation & interpretability, and finally deploying a usable Streamlit interface.

It’s not perfect — and it’s still improving — but it works, and it genuinely represents my learning, problem-solving and persistence.

🚀 Key Features

Hybrid CNN + ResNet50 architecture

~141,000 real & fake facial images collected and cleaned manually

Trained with callbacks, checkpoints & cosine learning schedulers

GPU-accelerated training using TensorFlow on WSL2 + CUDA + cuDNN

Evaluation metrics: Confusion Matrix, ROC Curve, PR Curve

Grad-CAM++ Interpretability visualization

Streamlit UI for real-time testing

Fully reproducible, modular project structure

📈 Model Performance

                               Metric	                               Result
                        Validation Accuracy                        	~99.9%
                        Validation Loss	                           ~0.009
                        Training Platform                     	WSL2 + CUDA-accelerated GPU
                        Evaluation Insights	          Confusion Matrix, ROC & PR curves were near-perfect
                        

💡 Such extremely high accuracy likely indicates dataset imbalance or limited variability
Current work focuses on improving robustness and generalization to real-world deepfakes.

🧱 Tech Stack

                     Category	                              Tools Used
                     Dataset                 	~141k real & fake images (balanced 50-50)
                     Model                          	Hybrid CNN + ResNet50
                   Frameworks                       	TensorFlow 2.15, Keras
                   GPU & System                      	WSL2, CUDA 11.8, cuDNN 8.9
                   Evaluation                      	sklearn, matplotlib, seaborn
                     Deployment                           Streamlit UI
                  Visualization                            Grad-CAM++

                  
🗂 Project Structure
DEEPFAKE-IMAGE-DETECTION
│—— Final_dataset/
│—— models/
│   └—— build_hybrid_CNN.py
│   └—— train.py
│   └—— checkpoints/
│       └—— initial/best_model.h5
│       └—— fine-tune/best_model.h5
│—— evaluation/
│   └—— confusion_matrix.py
│   └—— roc_curve.py
│   └—— pr_curve.py
│—— visualization/
│   └—— gradcam_pp.py
│—— app.py
│—— detection.py
│—— README.md

▶️ Running the Application
Run Streamlit App
streamlit run "PATH_TO/app.py"

Predict Manually
python detection.py --image /path/to/image.png

🛠 Improvements in Progress

    1. Larger & more diverse real-world dataset

    2. Adversarial robustness testing

    3. Better Grad-CAM++ interpretability

    4. Temporal / video-based deepfake detection

    5. ViT-based experimentation

📚 What I Learned

    1. Setting up CUDA + cuDNN + TensorFlow on WSL2 was harder than model training itself 😅

    2. Accuracy isn’t everything — robustness matters more

    3. Interpretability is essential, not optional

    4. Persistence > Talent

    5.Dataset work = cleaning, balancing & understanding — not just downloading

    6. Machine Learning is engineering + science + patience

💛 Built With

Curiosity · Consistency · Persistence
I build, I experiment, I learn.

👤 Author

Anushka Verma
B-Tech CSE (AI/ML) | Dr. A.P.J Abdul Kalam Technical University (AKTU)
Machine Learning • Full-Stack AI Developer • Deep Learning Enthusiast

📩 Email — anuushka27@gmail.com

🌐 Portfolio — Coming Soon







