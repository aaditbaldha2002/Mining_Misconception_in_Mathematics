# EEDI - Mining Misconceptions in Mathematics

**Competition Link:** [https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics)

## Project Overview

This project implements a solution for the [EEDI - Mining Misconceptions in Mathematics](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics) Kaggle competition. The goal is to develop an NLP model that accurately predicts the relationship between misconceptions and incorrect answers (distractors) in multiple-choice mathematics questions.

## Problem Statement

In educational settings, multiple-choice questions use distractors to capture specific misconceptions that students might have. Tagging these distractors with appropriate misconceptions helps educators understand and address student learning gaps, but this process is time-consuming and inconsistent when done manually. This project aims to automate and improve this process.

## Dataset Description

The competition dataset consists of:

- **questions.csv**: Contains multiple-choice math questions with IDs, subjects, constructs, and text
- **misconceptions.csv**: Lists misconceptions with IDs and descriptions
- **train.csv**: Maps questions to misconceptions (the training labels)
- **test.csv**: Questions for which we need to predict misconceptions

Each question has one correct answer and three distractors (incorrect answers). The model's task is to predict which misconceptions are associated with each distractor.

## Model Architecture

The solution implements a two-stage approach:

1. **Dual Encoder Model**:
   - Uses two separate encoders for questions and misconceptions
   - Learns embeddings in the same vector space
   - Enables efficient similarity search
   - Based on transformer architecture (MPNet)

2. **Reranking Model**:
   - Cross-encoder model for refining initial predictions
   - Considers full context of both question and misconception
   - Improves ranking of retrieved misconceptions

## Evaluation Metric

The competition uses **Mean Average Precision @ 25 (MAP@25)** as the evaluation metric, which measures how well the model ranks relevant misconceptions within the top 25 predictions.

## Project Structure

```
.
├── Data/                      # Data directory
│   ├── train.csv             # Training data
│   ├── test.csv              # Test data
│   ├── misconceptions.csv    # Misconception descriptions
│   └── sample_submission.csv # Sample submission format
├── eedi_missconception.py    # Main implementation file
├── requirements.txt          # Project dependencies
└── README.md                # Project documentation
```

## Setup and Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd NLP-Project
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. **Important**: Before running any project code, ensure all dependencies are properly installed:
```bash
pip install -r requirements.txt
```

## Usage

1. **Data Preparation**:
   - Place the competition data files in the `Data/` directory
   - Run the preprocessing scripts to prepare the data

2. **Training**:
   - Train the dual encoder model:
   ```python
   python eedi_missconception.py --mode train
   ```
   - Train the reranker model:
   ```python
   python eedi_missconception.py --mode rerank
   ```

3. **Inference**:
   - Generate predictions:
   ```python
   python eedi_missconception.py --mode predict
   ```

## Model Performance

The solution achieves competitive performance through:
- Efficient dual encoder architecture
- Careful data preprocessing and augmentation
- Cross-validation strategy
- Ensemble methods for final predictions

## Dependencies

See `requirements.txt` for a complete list of dependencies. Key packages include:
- PyTorch
- Transformers
- Pandas
- NumPy
- Scikit-learn

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Project Contributors

- Aadit Baldha – abaldha@stevens.edu – CWID: 20029691
- Harsha Dasari – hdasari@stevens.edu – CWID: 20030116
- Rishi Chhabra – rchhabra1@stevens.edu – CWID: 20034068

## Acknowledgments

- EEDI competition organizers
- Hugging Face for transformer models
- PyTorch team for the deep learning framework