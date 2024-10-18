# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

# _My Image Classifier_

## Description
This project is an image classifier that uses deep learning to categorize flower images into 102 different categories. It was developed as part of the final project for the AI Programming with Python Nanodegree Program by Udacity.

## Prerequisites
- Python 3.11
- PyTorch
- Torchvision
- Matplotlib
- NumPy
- Pillow
For specific dependencies refere to REQUIREMENTS.TXT

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/your-repository.git
    cd your-repository
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # macOS/Linux
    .\venv\Scripts\activate  # Windows
    ```

3. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
### Training the Model
To train the model, run:
```bash
python train.py --data_dir <path/to/data/folder>
or to test it as it is
python train.py --model_name vgg16 --lr 0.001 --epochs 5 --hidden_units 128
```
### Inference
To make predictions with the trained model, run:
```bash
python predict.py --checkpoint <path/to/checkpoint.pth> --test_dir <path/to/test/folder>
```
## Contributing
The opportunity to carry out this project was provided to me by AWS, sponsoring my training through the "Udacity Nanodegree Program Python for IA + ML". During these months, I acquired several skills that were implemented in this project. And I still want to improve it further, potentially turning it into a smartphone or other device app.

In part 1, which is in .html format, I aimed for the code to be just functional. In part 2, I sought to refine the redundant and/or unnecessary parts to build a lean, reproducible, and feasible model. There are still corrections to be made, but the primary goal was to acquire the competence and technical skills to make the solution possible.

Feel free to open issues and pull requests.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contact

GitHub: Fabicca86

