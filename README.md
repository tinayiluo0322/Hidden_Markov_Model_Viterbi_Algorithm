### POS Tagger Prediction Using Hidden Markov Model

#### Description
This project develops a Part-of-Speech tagger using a Hidden Markov Model. It utilizes the first 10,000 tagged sentences from the Brown corpus to generate the components of the model: the transition matrix, observation matrix, and initial state distribution. The tagger is then tested on sentences 10150-10152 of the Brown corpus, comparing the inferred sequence of states against the truth. 

#### Features
- Implementation of a Part-of-Speech tagger using a Hidden Markov Model.
- Generation of transition matrix, observation matrix, and initial state distribution using the Brown corpus.
- Accommodation for Out-Of-Vocabulary words and smoothing.
- Testing and accuracy assessment using sentences from the Brown corpus.
- Detailed analysis and explanation of correct and incorrect POS predictions.

#### Technologies Used
- Python 3.11.5
- NLTK Library
- NumPy

## Code Overview

This code effectively implements a POS tagger using a Hidden Markov Model, generates essential matrices for the model, applies the Viterbi algorithm for tag prediction, and evaluates the model's performance on a subset of the Brown corpus.

- `getWordTag(corpus)`: Extracts and returns unique words and tags from the provided corpus, including an 'OOV' (Out-Of-Vocabulary) token to handle words not seen during training.

- `genMatrix(corpus, word_map, tag_map)`: Generates the initial state distribution matrix (`initState`), the transition matrix (`matState`), and the observation matrix (`matObs`) for the Hidden Markov Model. All matrices are normalized and initialized with Laplace smoothing.

- `viterbi(obs, pi, A, B)`: This function is the core of the POS tagging implementation, applying the Viterbi algorithm to infer the most likely sequence of states (tags) for a given sequence of observations (words). It handles observed values, initial state probabilities, state transition probabilities, and emission probabilities.

- `words_to_indices(words, word_map)`: Converts a list of words into their corresponding indices as per the `word_map`, substituting unknown words with the "OOV" index.


## Testing and Evaluation

- The POS tagger was rigorously tested using sentences 10150-10153 from the Brown corpus, accessed via NLTK's `universal_tagset`.

- The Viterbi algorithm was applied to each test sentence for predicting POS tags, leveraging the model trained on the first 10,000 sentences of the same corpus.

- Accuracy was calculated by comparing the predicted tags against the actual tags in the test data. Detailed results, including test sentences, predicted tags, actual tags, and accuracy percentages, were printed to evaluate the model's performance.

- Overall accuracy was computed to provide a quantitative measure of the tagger's performance across the tested sentences.


### General Observation
The POS tagger achieved an overall accuracy of 93.62%, indicating that the model correctly predicted the POS tags in the majority of cases. This level of accuracy is commendable, given the complexities of human language and the inherent limitations of the training set.

### Correct POS Prediction Explanation
The success of the model can be attributed to the diverse training data and the accurate implementation of matrix-generation methods and the Viterbi algorithm in the HMM.

### Incorrect POS Prediction Explanation
- **First Sentence (Accuracy: 92.31%):** The word "coming" was predicted as 'NOUN' but was actually a 'VERB'. This misclassification is due to the word's dual nature in the training data.
- **Second Sentence (Accuracy: 88.89%):** The word "face-to-face" was an OOV token, leading to a less accurate prediction. "Another" was incorrectly tagged as 'NOUN' instead of 'DET', possibly due to the model's bias towards frequent patterns.
- **Third Sentence (Accuracy: 100%):** The accurate predictions suggest a good representation of these words and contexts in the training data.

### Conclusion
The model's errors primarily stem from:
1. Ambiguity of certain words in the training data.
2. Bias towards more frequent patterns or contexts for similar words.
3. Absence of some words in the training data, leading to inaccurate generalizations for OOV tokens.
