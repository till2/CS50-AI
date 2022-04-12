import nltk
import sys

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> NP VP | NP VP NP | NP VP P NP | NP VP P Det NP | NP VP Det NP 
S -> Det NP VP | Det NP VP NP | Det NP VP P NP | Det NP VP P Det NP
S -> S Conj S | S Conj VP NP | S Conj VP Det NP
S -> NP VP Det NP | Det NP VP Det NP
NP -> N | Adj NP | NP P NP | NP Adv | NP P Det NP
VP -> V | Adv V | V Adv
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """

    tokens = [token.lower() for token in nltk.word_tokenize(sentence)
                    if any(char.isalpha() for char in token)]

    return tokens


def has_np_child_chunks(tree):
    """
    Helper function that determines if a tree has
    noun phrase chunks in it's subtrees.
    """

    labels = [s.label() for s in tree.subtrees() if s is not tree]

    for label in labels:
        if label == "NP":
            return True

    return False


def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """

    np_chunks = []

    for s in tree.subtrees():
        if s.label() == "NP" and s not in np_chunks and not has_np_child_chunks(s):
            np_chunks.append(s)

    # return list of noun-phrase tree objects
    return np_chunks


if __name__ == "__main__":
    main()
