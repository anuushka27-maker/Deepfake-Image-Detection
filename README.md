Hybrid CNN + ResNet50 | Grad-CAM++ | Streamlit UI | ROC | PR | Confusion Matrix 
This Project is an end-to-end system for detecting Real VS Fake Images Deep-Learning. I built everything from scratch - Dataset Collection of ~141K Real & Fake Images, Dataset Clean-UP, Model Design, Training(GPU accelerating on WSL2), Interpretability Visualizations, and UI Deployment.
It's not perfect, and it's still improving - but it works and it reflects my learning journey.

FEATURES

1. HYBRID CNN-ResNet50 architecture
2. Trained on ~141k real and fake face images collected from multiple sources
3. Fine-tuned with Callbacks, Checkpoints & Learning Schedulers
4. GPU-accelerated training using WSL2+CUDA+cuDNN
5. Evaluation Metrics:
   a. Confusion Matrix
   b. ROC Curve
   c. PR Curve
6. Streamlit UI
7. Grad-CAM++ visualization(interpretability)

MODEL PERFORMANCE

         METRIC                     RESULT                                  
    Validation Accuracy             ~99.9%
    Validation Loss                 ~0.009
    Training PLatform               WSL2 GPU with Tensorflow

Confusion Matrix, ROC Curve, PR Curve are too perfect.

Accuracy looks extremely high high - might indicate Dataset quality issues(Fake images much easier than real-world case).
Working on improving generalisations & robustness.

TECH STACK

          CATEGORY                             TOOLS
          Dataset                            141,000 Real & Fake Images(~50-50%) 
          Model                              RestNet50 + Hybrid CNN
          Frameworks                         Tensorflow 2.15, Keras
          GPU                                WSL2, CUDA 11.8, cuDNN 8.9
          Evaluation                         sklearn, matplotlib, seaborn
          UI                                 Streamlit
          
PROJECT STRUCTURES

DEEPFAKE-IMAGE-DETECTION
            |___________Final_dataset
            |___________models/
            |               |______build_hybrid_CNN.py
            |
            |___________train.py
            |
            |___________checkpoints/
            |               |______initial/
            |               |         |____best_model.h5
            |               |______fine-tune/
            |                         |_____best_model.h5
            |___________evaluation/
            |               |______confusion_matrix.py
            |               |______roc-curvepy
            |               |______pr_curve.py
            |
            |___________visualisation
            |               |______gradcam_pp.py
            |___________app.py
            |
            |___________ README.md

RUNNING THE APP

------------------In VS Code WSL2 terminal run streamlit run "PATH_TO_app.py"---------------------

PREDICT MANUALLY

-----------------python detection.py --image /path/to/image.png----------------------------------

IMPROEMENT IN PROGRESS 

1. More real-World Dataset
2. Adversarial testing
3. Better Grad-CAM++ results
4. Video-Based Deepfake Detection

MY LEARNING TAKEAWAYS

1. Setting Up GPU + CUDA + WSL2 was more harder than Training and Dataset collecting and cleanup
2. Accuracy isn't everything-robustness matters more
3. Interpretability matters in DeepLearning
4. Debugging and Persistence is the Real Skills
5. Dataset handling isn't just Downloading or collecting RAW. 
6. Tensorflow is very Messy.
7. Machine Learning isn't Just Training

BUILT WITH

CURIOSITY . CONSISTENCY . PERSISTENCE 
I build, I experiment, I learn.

AUTHOR

Anushka
B-Tech CSE(AI/ML), Dr. A.P.J Abdul Kalam Technical University(AKTU)
Machine Learning Full Stack AI Enthusiast

Portfolio Coming Soon

E-mail:- anuushka27@gmail.com







