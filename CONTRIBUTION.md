# Contribution Statement

## Team Name
Emoji Intelligence Unit

## Course
COMP6713 — Natural Language Processing, UNSW T1 2026

---

## Team Members

| Name | zID |
|---|---|
| Samyak Diwan | z5611048 |
| Gurdiraj Bal | z5386590 |
| To Sang Ngan | z5342080 |
| Junwei Wang | z5328887 |

---
All team members contributed equally to this project. Throughout the development process, we supported one another by leaning into each person's strengths whether in research, coding, design, or writing. When a teammate faced other commitments, the rest of the group stepped up to cover their areas without hesitation. Every major decision was made collaboratively, and no single member carried a disproportionate share of the workload. Our collective effort and mutual flexibility were essential to completing this project successfully

## Individual Contributions

### Samyak Diwan (z5611048)

- **Stage 1 Data Pipeline:** Downloaded and standardised the TweetEval and SARCdatasets; built the ESR emoji feature dictionary; implemented the three preprocessing modes (T/K/D) applied across all stages.
- **Stage 3 Fine-tuning:** Fine-tuned `cardiffnlp/twitter-roberta-base` on TweetEval under all three emoji modes; managed training loop, evaluation on TweetEval test set, and checkpoint saving.
- **Stage 4 Error Analysis:** Identified and categorised 10 misclassified examples from the best model; produced the error analysis CSV with labelled error types.
- **Stage 5 CLI (`predict.py`):** Implemented the inference script supporting single-text and file-based input, all three emoji modes, configurable threshold, and ESR feature reporting.
- **README.md:** Implemented the README file.
- Organised weekly and in person meetings. Allocated work to team-members.

---

### Gurdiraj Bal (z5386590)
- **Stage 2:** Run RoBERTa inference model
- **Stage 3:** Tried fine tuning on RoBERTa
- Worked on parts c & d of the final report 
- Worked on final presentation

---

### To Sang Ngan (z5342080)
- Compile and merge part and part 2 code into one single pipeline
- handle and develop the tfidf baseline model with evaluation and also extra emoji sentiments ranking feature with baseline model to compare the performance differences
- handle the introduction including problem definition and the domain
Of the report . Also including the model part for tfidf and the baseline model with evaluation and explanation of the model structure 
- handle the slides designing and also schedule group meeting while coordinating other work

### Junwei Wang (z5328887)
- **Stage 2 Pretrained model:** test pretrained model bertweet firstly on SARC 2.0 and tweetEval, propose to replace SARC 2.0 with iSarcasm due to poor performance.
- **Stage 2 literature review:** review how the pretrained models were trained and how the dataset were anotated
- **Stage 2 Evaluation pretrained result:** identified the recall and precision trade off of two pretrained models, plots for better visualization
- Presentation slides: add few slides for dataset, Stage 2 and 3

---