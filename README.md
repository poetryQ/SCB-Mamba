# SCB-Mamba
A Weight Sharing Mamba with Super pixel Clustering and Boundary Correction for Endometrial Injury Severity Classification in OCT Images.
<img width="866" height="434" alt="image" src="https://github.com/user-attachments/assets/2d7e31d3-6482-4ef3-9f7d-f86675e72b83" />
- **Advanced Architecture**: Combines Mamba models with superpixel clustering and boundary correction
- **Multi-path Weight Sharing**: Efficient feature extraction with shared parameters
- **Comprehensive Evaluation**: Multiple metrics and visualization tools
# Quick Start
## Data Preparation
data/  
├── normal/  
│   ├── image1.png  
│   ├── image2.png  
│   └── ...  
├── mild/  
│   ├── image1.png  
│   └── ...  
└── severe/  
    ├── image1.png  
    └── ...  
## Training
python main.py --mode train --data_dir ./data --epochs 100 --batch_size 32
## Testing
python main.py --mode test --data_dir ./data --checkpoint ./checkpoints/best_model.pth
## Inference
python main.py --mode inference --image_path ./sample.jpg --checkpoint ./checkpoints/best_model.pth
## Batch inference:
python main.py --mode inference --image_dir ./test_images --checkpoint ./checkpoints/best_model.pth

# Model Architecture
The SCB-Mamba model consists of three main components:  

GSDE Module: Grayscale Saliency Difference Enhancement using superpixel clustering  
<img width="839" height="404" alt="image" src="https://github.com/user-attachments/assets/f03ab508-7baa-474b-ab70-d84598cacdb8" />

IRBE Module: Injured Region Boundary Enhancement with spatial alignment  
<img width="866" height="183" alt="image" src="https://github.com/user-attachments/assets/2c2b6b70-1065-4c32-b38a-3c468bfc2eb6" />


Multi-path Mamba Backbone: Weight-sharing Mamba architecture with multiple scanning strategies  

<img width="866" height="341" alt="image" src="https://github.com/user-attachments/assets/d766bb77-14d0-42d3-9b75-6e1416579257" />

