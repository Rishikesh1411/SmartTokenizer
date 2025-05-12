import nltk
from nltk.tokenize import word_tokenize, sent_tokenize, TreebankWordTokenizer, RegexpTokenizer
from nltk.corpus import stopwords

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
# Download the 'punkt_tab' data package
nltk.download('punkt_tab', quiet=True)  # This line is added to download the missing data package

def tokenize_words(text, remove_punctuation=False, remove_stopwords=False):
    """Tokenizes words with options to remove punctuation and stopwords."""
    tokens = word_tokenize(text)
    if remove_punctuation:
        tokens = [token for token in tokens if token.isalnum()]
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token.lower() not in stop_words]
    return tokens

def tokenize_sentences(text, min_length=0):
    """Tokenizes sentences with an option to filter short sentences."""
    sentences = sent_tokenize(text)
    return [sent for sent in sentences if len(sent.split()) >= min_length] if min_length > 0 else sentences

def tokenize_treebank(text):
    """Tokenizes words using the Treebank tokenizer."""
    return TreebankWordTokenizer().tokenize(text)

def tokenize_regex(text, pattern=r'\w+'):
    """Tokenizes words using a regex pattern."""
    return RegexpTokenizer(pattern).tokenize(text)

# Updated sample text about Rishikesh, IIT Patna, and data science
sample_text = (
    "Rishikesh is a beautiful city located in the foothills of the Himalayas. "
    "IIT Patna offers a comprehensive program in Data Science, Artificial Intelligence, and Machine Learning. "
    "Students at IIT Patna engage in cutting-edge research and projects that focus on real-world applications. "
    "The curriculum is designed to equip students with the necessary skills to excel in the tech industry."
)

# Tokenization results
print("Word Tokens:", tokenize_words(sample_text, remove_punctuation=True, remove_stopwords=True))
print("Sentence Tokens:", tokenize_sentences(sample_text, min_length=3))
print("Treebank Tokens:", tokenize_treebank(sample_text))
print("Regex Tokens:", tokenize_regex(sample_text))


