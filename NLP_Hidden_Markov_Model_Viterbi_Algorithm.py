"""Viterbi Algorithm for inferring the most likely sequence of states from an HMM.
Patrick Wang, 2021
"""

from typing import Sequence, Tuple, TypeVar
import numpy as np
import nltk

# nltk.download("brown")
# nltk.download("universal_tagset")


def getWordTag(corpus):
    words = set()
    tags = set()
    for sentence in corpus:
        for word, tag in sentence:
            words.add(word)
            tags.add(tag)
    # Add OOV token to word list
    words.add("OOV")
    # Convert sets to lists for index-based operations
    word_map = list(words)
    tag_map = list(tags)
    return word_map, tag_map


def genMatrix(corpus, word_map, tag_map):
    num_tags = len(tag_map)
    num_words = len(word_map)
    # Create dictionaries for efficient index lookup
    word_to_idx = {word: idx for idx, word in enumerate(word_map)}
    tag_to_idx = {tag: idx for idx, tag in enumerate(tag_map)}
    # 1. Initial State Distribution Matrix
    initState = np.ones(num_tags)  # Laplace Smoothing
    for sentence in corpus:
        _, tag = sentence[0]  # consider only the first word of each sentence
        initState[tag_to_idx[tag]] += 1
    initState /= initState.sum()  # Normalize
    # 2. Transition Matrix
    matState = np.ones((num_tags, num_tags))  # Laplace Smoothing
    for sentence in corpus:
        for i in range(len(sentence) - 1):
            curr_tag_idx = tag_to_idx[sentence[i][1]]
            next_tag_idx = tag_to_idx[sentence[i + 1][1]]
            matState[curr_tag_idx][next_tag_idx] += 1
    # Normalize each row
    matState /= matState.sum(axis=1, keepdims=True)
    # 3. Observation Matrix
    matObs = np.ones((num_tags, num_words))  # Laplace Smoothing
    for sentence in corpus:
        for word, tag in sentence:
            matObs[tag_to_idx[tag]][word_to_idx[word]] += 1
    # Normalize each row
    matObs /= matObs.sum(axis=1, keepdims=True)
    return matState, matObs, initState


# Example Usage:

training = nltk.corpus.brown.tagged_sents(tagset="universal")[:10000]
word_map, tag_map = getWordTag(training)
matState, matObs, initState = genMatrix(training, word_map, tag_map)
# print("Initial State (initState):", initState)
# print("Transition Matrix (matState):", matState)
# print("Observation Matrix (matObs):", matObs)
# Q represents the states
# V represents the observations

Q = TypeVar("Q")
V = TypeVar("V")
# obs: A sequence of observed values (usually word indices).
# pi: Initial state probabilities.
# A: State transition probabilities.
# B: Emission probabilities.
# It returns a tuple containing the most probable sequence of states (qs) and its probability.


def viterbi(
    obs: Sequence[int],
    pi: np.ndarray[Tuple[V], np.dtype[np.float_]],
    A: np.ndarray[Tuple[Q, Q], np.dtype[np.float_]],
    B: np.ndarray[Tuple[Q, V], np.dtype[np.float_]],
) -> tuple[list[int], float]:
    """Infer most likely state sequence using the Viterbi algorithm.
    Args:
        obs: An iterable of ints representing observations.
        pi: A 1D numpy array of floats representing initial state probabilities.
        A: A 2D numpy array of floats representing state transition probabilities.
        B: A 2D numpy array of floats representing emission probabilities.
    Returns:
        A tuple of:
        * A 1D numpy array of ints representing the most likely state sequence.
        * A float representing the probability of the most likely state sequence.
    """
    # N: The number of observations.
    # Q: The number of states.
    # V: The number of unique observations.
    N = len(obs)
    Q, V = B.shape  # num_states, num_observations
    # d_{ti} = max prob of being in state i at step t
    #   AKA viterbi
    # \psi_{ti} = most likely state preceeding state i at step t
    #   AKA backpointer
    # initialization
    log_d = [np.log(pi) + np.log(B[:, obs[0]])]
    log_psi = [np.zeros((Q,))]
    # recursion
    for z in obs[1:]:
        log_da = np.expand_dims(log_d[-1], axis=1) + np.log(A)
        log_d.append(np.max(log_da, axis=0) + np.log(B[:, z]))
        log_psi.append(np.argmax(log_da, axis=0))
    # termination
    log_ps = np.max(log_d[-1])
    qs = [-1] * N
    qs[-1] = int(np.argmax(log_d[-1]))
    for i in range(N - 2, -1, -1):
        qs[i] = log_psi[i + 1][qs[i + 1]]
    return qs, np.exp(log_ps)


def words_to_indices(words, word_map):
    """Convert a list of words to their respective indices."""
    return [
        word_map.index(word) if word in word_map else word_map.index("OOV")
        for word in words
    ]


# Extracting test data

test_data = nltk.corpus.brown.tagged_sents(tagset="universal")[10150:10153]
# Function to compute accuracy


def accuracy(predicted, actual):
    return sum(p == a for p, a in zip(predicted, actual)) / len(predicted)


total_correct_tags = 0
total_tags = 0
# Looping over each sentence in the test data

for test_sentence in test_data:
    words = [
        word for word, _ in test_sentence
    ]  # Extracting words from the (word, tag) tuples
    obs_indices = words_to_indices(words, word_map)  # Convert words to indices
    state_sequence, _ = viterbi(obs_indices, initState, matState, matObs)
    predicted_tags = [tag_map[idx] for idx in state_sequence]
    actual_tags = [tag for _, tag in test_sentence]
    total_correct_tags += sum(p == a for p, a in zip(predicted_tags, actual_tags))
    total_tags += len(actual_tags)
    acc = accuracy(predicted_tags, actual_tags)
    print("Test sentence:", words)
    print("Predicted tags:", predicted_tags)
    print("Actual tags:", actual_tags)
    print("Accuracy:", acc)
    print("Accuracy in Percentage:", round(acc * 100, 2), "%")
    print("\n" + "=" * 50 + "\n")

overall_accuracy = total_correct_tags / total_tags
print(f"Overall Accuracy for the three sentences: {overall_accuracy}")
print(
    f"Overall Accuracy for the three sentences in percentage: {overall_accuracy * 100:.2f}%"
)
