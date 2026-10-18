# Predicting poll scores

First Pipeline assignment: predict an instructor-assigned score of 2, 3, or 4 from course, poll question, and the student's response.

## Dataset

[graded_polls.csv](data/graded_polls.csv) contains **137 personal graded responses across 12 courses**, collected from Forum using Codex computer use. This is the October 18, 2026 snapshot used in the preliminary analysis: 16 twos, 66 threes, and 55 fours. The [manifest](data/manifest.json) records counts, transformations, and file hashes.

| Column | Meaning | Role |
| --- | --- | --- |
| response_id | Collector's stable poll/grade record identifier | Traceability; exclude from model inputs |
| course | Course code | Input |
| question | Poll question text | Input |
| response | Student response text | Input |
| score | Recorded grade: 2, 3, or 4 | Target |
| session_number | Session identifier within the course | Grouping/audit; exclude from model inputs |
| grading_tag | Participation or subject-specific rubric tag | Label audit only; exclude from model inputs |

The published CSV preserves the source values and all rows while selecting and renaming these seven columns. Instructor feedback, Forum URLs, dates/times, and descriptive titles are omitted. Encoding, token-length handling, and train/validation/test splitting have not been applied to this CSV. No examples have been duplicated to balance labels.

## Additional dataset: Student 2

[minerva_graded_polls_student_2.csv](data/minerva_graded_polls_student_2.csv) contains 123 graded poll records from Student 2 in a separate file with all 14 original columns. It has not been merged into the 137-response dataset or included in the preliminary analysis. Unlike the seven-column dataset above, this export includes instructor feedback and source metadata, which should be excluded from predictive model inputs.

## Planned comparison

All three conditions use the same frozen MiniLM embeddings: multinomial logistic regression, an eight-unit feedforward neural network, and the same network with dropout. Use shared five-fold score-stratified evaluation, with validation inside the training portion. The simplified implementation is still being planned; code and notebooks are not included in this data-only update.

## Limitations

The dataset includes participation and subject-specific rubric grades; equivalent numeric scores may represent different grading criteria. It is a small, imbalanced archive from one student, and recorded grades do not establish objective answer quality. Grading tags and instructor feedback must not be used to predict a score. Earlier exploratory modeling has already inspected the dataset, so it is not an untouched final test resource.
