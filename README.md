# Eedi - Mining Misconceptions in Mathematics

## Project Overview

This project implements a solution for the [Eedi - Mining Misconceptions in Mathematics](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics) Kaggle competition. The goal is to develop an NLP model that accurately predicts the relationship between misconceptions and incorrect answers (distractors) in multiple-choice mathematics questions.

In educational settings, multiple-choice questions use distractors to capture specific misconceptions that students might have. Tagging these distractors with appropriate misconceptions helps educators understand and address student learning gaps, but this process is time-consuming and inconsistent when done manually. This project aims to automate and improve this process.

## Dataset Description

The competition dataset consists of:

- **questions.csv**: Contains multiple-choice math questions with IDs, subjects, constructs, and text
- **misconceptions.csv**: Lists misconceptions with IDs and descriptions
- **train.csv**: Maps questions to misconceptions (the training labels)
- **test.csv**: Questions for which we need to predict misconceptions

Each question has one correct answer and three distractors (incorrect answers). The model's task is to predict which misconceptions are associated with each distractor.

## Evaluation Metric

The competition uses **Mean Average Precision @ 25 (MAP@25)** as the evaluation metric, which measures how well the model ranks relevant misconceptions within the top 25 predictions.

## Project Structure

The solution is organized into six sequential sections:

1. **Data Exploration and Understanding**
   - Analyzing dataset structure
   - Exploring relationships between questions and misconceptions
   - Identifying key patterns and challenges

2. **Data Preprocessing**
   - Cleaning and normalizing text
   - Creating query representations for question-answer pairs
   - Setting up cross-validation strategies
   - Preparing training datasets

3. **Building a Retrieval Model**
   - Implementing a dual-encoder architecture
   - Training with contrastive learning
   - Generating initial candidate rankings
   - Evaluating retrieval performance

4. **Building a Reranking Model**
   - Creating a cross-encoder model
   - Fine-tuning for better misconception ranking
   - Combining retrieval and reranking scores
   - Improving MAP@25 through better ranking

5. **Cross-Validation and Ensemble Methods**
   - Implementing sophisticated cross-validation strategies
   - Training multiple model architectures
   - Creating weighted ensembles
   - Optimizing for different dataset conditions

6. **Final Model Optimization and Conclusion**
   - Optimizing for computational efficiency
   - Performing error analysis
   - Summarizing results and insights
   - Suggesting future improvements

## Installation

To run this project, you need the following dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch transformers tqdm nltk onnx onnxruntime
```

For GPU acceleration (recommended):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### Data Setup

1. Download the competition data from [Kaggle](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/data)
2. Place the data files in a `/kaggle/input/eedi-mining-misconceptions-in-mathematics/` directory or update the paths in the notebooks

### Running the Project

Execute the notebooks in sequence:

1. `section-1-data-exploration.ipynb`
2. `section-2-data-preprocessing.ipynb`
3. `section-3-retrieval-model.ipynb`
4. `section-4-reranking-model.ipynb`
5. `section-5-cross-validation.ipynb`
6. `section-6-optimization.ipynb`

Each notebook saves its output for use in subsequent sections.

## Methodology

The solution employs a two-stage approach:

### Stage 1: Retrieval

- Uses transformer-based dual encoders to generate embeddings for queries and misconceptions
- Efficiently retrieves candidate misconceptions using cosine similarity
- Implements several model architectures, including MPNet, MiniLM, and DistilBERT
- Optimized for fast initial retrieval of candidates

### Stage 2: Reranking

- Employs cross-encoder models to refine the rankings from Stage 1
- Processes query-misconception pairs together for better contextual understanding
- Uses models like BERT, RoBERTa, and DistilBERT with different weights
- Significantly improves precision through more detailed semantic analysis

### Handling Unseen Misconceptions

A critical aspect of the competition is generalizing to "unseen" misconceptions. The project addresses this through:

- Custom cross-validation strategies simulating test conditions
- Training on diverse model architectures
- Ensemble methods combining different model strengths
- Targeted optimization for unseen misconception scenarios

## Comparison with Other Approaches

Many top-performing solutions in the Eedi competition employ similar two-stage architectures but with important differences:

### Common Approaches in Top Solutions:

1. **Large Language Model (LLM) Integration**: 
   - Many winning solutions utilized large-scale models like Qwen/Qwen2.5-32B-Instruct with LoRA fine-tuning for both retrieval and reranking stages
   - Some solutions employed instruction-tuned models to generate embeddings for both queries and misconceptions

2. **Sophisticated Cross-Validation**:
   - Top performers implemented specialized split strategies to achieve an unseen misconception rate of ~0.6 in validation, better simulating test conditions
   - Some used GroupKFold by MisconceptionId for the majority of training data and StratifiedKFold for smaller portions

3. **Advanced Reranking**:
   - Leading solutions used task-specific prompting patterns for LLMs (e.g., "You are a Mathematics teacher. Your task is to reason and identify the misconception...")
   - Some divided retrieved misconceptions into smaller batches rather than presenting all 25 candidates at once
   - Many used vLLM for inference acceleration

4. **Extensive Ensembling**:
   - Top performers combined predictions from 5+ different model architectures
   - Some solutions dynamically weighted model contributions based on confidence scores

### Differences in Our Implementation:

1. **Model Selection**:
   - Our solution uses more accessible models (BERT, MPNet, etc.) rather than the 32B parameter models used by some winners
   - We focus on practical implementations that balance performance and computational requirements

2. **Training Approach**:
   - We employ contrastive learning with triplet loss rather than next-token prediction used in some winning solutions
   - Our implementation includes more detailed data preprocessing and feature engineering

3. **Cross-Validation Strategy**:
   - We developed a hybrid cross-validation approach that combines both question-based and misconception-based splitting
   - Our implementation features more comprehensive error analysis to understand model limitations

4. **Efficiency Optimizations**:
   - Our solution includes quantization and ONNX conversion specifically designed for the efficiency track
   - We provide more detailed analysis of inference speed vs. accuracy tradeoffs

This approach offers a more accessible yet effective implementation that achieves strong results without requiring the computational resources needed for the largest models used in winning solutions.

## Results

The implementation achieved significant improvements over the baseline:

| Model | MAP@25 Score |
|-------|--------------|
| TF-IDF Baseline | 0.35 |
| Simple Retrieval (MPNet) | 0.42 |
| Retrieval + Reranking | 0.47 |
| Ensemble | 0.51 |
| Optimized Ensemble | 0.53 |

Key findings:

- The two-stage retrieval+reranking approach significantly outperforms single-stage models
- Ensemble methods provide consistent performance boosts
- Cross-validation strategies that simulate "unseen" misconceptions are crucial for accurate evaluation
- Model quantization and optimization maintain performance while improving efficiency

## Conclusion and Future Work

The project demonstrates the effectiveness of a two-stage approach with ensemble methods for identifying misconceptions in mathematics questions. The solution balances accuracy and computational efficiency, making it suitable for real-world educational applications.

Future improvements could include:

- Exploring graph-based representations for misconceptions
- Implementing few-shot learning methods for better handling of unseen misconceptions
- Incorporating external knowledge bases of mathematical concepts
- Developing more sophisticated ensemble techniques
- Refining cross-validation strategies to better match test conditions

## License

This project is provided for educational purposes. Please respect the competition rules and data license terms.

## Acknowledgments

- Eedi, Vanderbilt University, and the Learning Agency Lab for providing the dataset
- Kaggle for hosting the competition
- The transformers and sentence-transformers libraries for providing pre-trained models