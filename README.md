# Mining_Misconception_in_Mathematics
This is project for the NLP Project for Stevens 2025 NLP coursework for the Batch of Fall 2024


File and Field Description:
- [train/test].csv
    - QuestionId - Unique question identifier (int).
    - ConstructId - Unique construct identifier (int) .
    - ConstructName - Most granular level of knowledge related to question (str).
    - CorrectAnswer - A, B, C or D (char).
    - SubjectId - Unique subject identifier (int).
    - SubjectName - More general context than the construct (str).
    - QuestionText - Question text extracted from the question image using human-in-the-loop OCR (str) .
    - Answer[A/B/C/D]Text - Answer option A text extracted from the question image using human-in-the-loop OCR (str).
    - Misconception[A/B/C/D]Id - Unique misconception identifier (int). Ground truth labels in train.csv; your task is to predict these labels for test.csv.
- misconception_mapping.csv - maps MisconceptionId to its MisconceptionName
- sample_submission.csv - A submission file in the correct format.
- QuestionId_Answer - Each question has three incorrect answers for which need you predict the MisconceptionId.
- MisconceptionId - You can predict up to 25 values, space delimited.
