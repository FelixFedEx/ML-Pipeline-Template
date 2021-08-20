#!/usr/bin/env python
# coding: utf-8

import re
import string
import unicodedata

import inflect
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer

import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('words')
# english_words = set(nltk.corpus.words.words())
# nltk.download("wordnet")


from duplicate_detection_model.processing.utils.hp_hexcode_parser import parse

"""These special tokens are splited from error names, because they are combination of characters and numbers. """
"""So we can choose to keep these words. """
no_replac_token_list = [
    "crash1",
    "phase1",
    "process1",
    "session2",
    "object1",
    "session3",
    "memory1",
    "session5",
    "usb3",
    "session4",
    "session1",
    "pp1",
    "io1",
    "win32k",
    "pp0",
    "security1",
    "hal1",
    "phase0",
]


def to_lowercase(urstring):
    """Convert all characters to lowercase from string"""
    return urstring.lower()


def to_halfwidth(ustring):
    """Convert characters from full-width to half-width"""
    ss = []
    for s in ustring:
        rstring = ""
        for uchar in s:
            inside_code = ord(uchar)
            if inside_code == 12288:  # convert full-width balnk to half-width
                inside_code = 32
            elif (
                inside_code >= 65281 and inside_code <= 65374
            ):  # convert all full-width characters to half-width.(Not include blank)
                inside_code -= 65248
            rstring += chr(inside_code)
        ss.append(rstring)
    return "".join(ss)


def remove_URL(urstring):
    """Remove URLs from a string"""
    #     return re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', ' ', urstring)
    return re.sub(
        r"(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b",
        " ",
        urstring,
        flags=re.MULTILINE,
    )


def to_decontract(urstring):
    """Restore abbreviations"""

    # specific
    urstring = urstring.replace("i/o", " io ")
    urstring = urstring.replace("%", " percent ")
    urstring = re.sub(r"won\'t", "will not", urstring)
    urstring = re.sub(r"can\'t", "can not", urstring)

    # general
    urstring = re.sub(r"n\'t", " not", urstring)
    urstring = re.sub(r"\'re", " are", urstring)
    urstring = re.sub(r"\'s", " is", urstring)
    urstring = re.sub(r"\'d", " would", urstring)
    urstring = re.sub(r"\'ll", " will", urstring)
    urstring = re.sub(r"\'t", " not", urstring)
    urstring = re.sub(r"\'ve", " have", urstring)
    urstring = re.sub(r"\'m", " am", urstring)

    return urstring


def parse_hp_hexcode(urstring):
    """Parse the error hex code to the real name"""
    return parse(urstring)


def remove_punctuation(urstring, remove_dot=False):
    """Remove punctuation from String"""
    ls_punc = list(string.punctuation)
    ls_punc.extend(["\r\n", "\r", "\n", "®", "“", "”", "(r)"])

    if remove_dot == False:
        ls_punc.remove(".")

    for pun in ls_punc:
        urstring = urstring.replace(pun, " ")

    return urstring


def remove_redundant_blank(urstring):
    """Remove redundant blanks from String"""
    urstring = re.sub(r"\s+", " ", urstring)
    urstring = urstring.strip()
    return urstring


def remove_isolated_character(words):
    """Remove only one characters from list of tokenized words"""
    new_words = []
    for word in words:
        word = remove_redundant_blank(word)
        if len(word) > 1:
            new_words.append(word)
    return new_words


def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words("english"):
            new_words.append(word)
    return new_words


def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = (
            unicodedata.normalize("NFKD", word)
            .encode("ascii", "ignore")
            .decode("utf-8", "ignore")
        )
        if len(new_word) > 1:
            new_words.append(new_word)
    return new_words


def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos="v")
        lemmas.append(lemma)
    return lemmas


def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems


def replace_integers2text(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words


def replace_number2blank(tokens, method_option=0, no_replace_code_name=True):

    """
    Description: Replace all numbers occurrences in list of tokenized words with special representation.
    Parameters:
            1.method_option : If set 0, start to replace number and dot-number to be blank.
                              If set 1, Start to replace to be string 'number'.

            2.no_replace_code_name : Contain the special words(HP hexcode name).
    """
    pattern_pure_dot = re.compile(r"^[0-9]+[\.][0-9]+$")
    pattern_pure_num = re.compile(r"^[0-9]+$")
    pattern_complex_num = re.compile(r"[0-9]+")

    if method_option == 0:
        for ix in range(len(tokens)):

            if (tokens[ix] in no_replac_token_list) and no_replace_code_name:
                continue

            if re.search(pattern_pure_dot, tokens[ix]):
                tokens[ix] = re.sub(pattern_pure_dot, " ", tokens[ix])

            elif re.search(pattern_pure_num, tokens[ix]):
                tokens[ix] = re.sub(pattern_pure_num, " ", tokens[ix])

            if re.search(pattern_complex_num, tokens[ix]):
                tokens[ix] = re.sub(pattern_complex_num, " ", tokens[ix])

        temp_str = " ".join(tokens)

        temp_str = remove_punctuation(temp_str, remove_dot=True)
        tokens = remove_isolated_character(temp_str.split(" "))

    elif method_option == 1:
        #     v3.5.6.7  => v_number_number_number
        #     3.4.5.6   => number_number_number_number
        #     win.6aws  => win_number_aws
        #     win9.6aws => win_number_number_was

        for ix in range(len(tokens)):

            if (tokens[ix] in no_replac_token_list) and no_replace_code_name:
                continue

            if re.search(pattern_pure_dot, tokens[ix]):
                tokens[ix] = re.sub(pattern_pure_dot, "number", tokens[ix])
                print(tokens[ix])
            elif re.search(pattern_pure_num, tokens[ix]):
                tokens[ix] = re.sub(pattern_pure_num, "number", tokens[ix])

            if re.search(pattern_complex_num, tokens[ix]):
                tokens[ix] = re.sub(pattern_complex_num, "_number_", tokens[ix])
                tokens[ix] = re.sub("_._", "_", tokens[ix])

            tokens[ix] = re.sub(r"^[.]+", "", tokens[ix])
            tokens[ix] = re.sub(r"[.]+$", "", tokens[ix])
            tokens[ix] = re.sub(r"[.]+", "", tokens[ix])
            tokens[ix] = re.sub(r"^[\_]+", "", tokens[ix])
            tokens[ix] = re.sub(r"[\_]+$", "", tokens[ix])
            tokens[ix] = re.sub(r"[\_]+", r"_", tokens[ix])

        temp_str = " ".join(tokens)
        temp_str = temp_str.replace(".", " ")
        tokens = remove_isolated_character(temp_str.split(" "))

    return tokens


def process_string(
    str,
    toLowercase=True,
    toHalfwidth=True,
    removeURL=True,
    toDecontract=True,
    parseHpHexcode=True,
    removePunctuation=True,
    removeDot=False,
    removeRedundantBlank=True,
):
    """
    Parameters:
        1. toLowercase  : Start to convert to lowercase letters.
        2. toHalfwidth  : Start to convert to halfwidth letters.
        3. removeURL    : Start to remove URL from string.
        4. toDecontract : Start to decontract abbreviations.
        5. parseHpHexcode : Start to replace integer to text representation.(Just only include positive integers.)
        6. removePunctuation : Start to remove puncations.
        6-1. removeDot : Choose to remove dot symbol from string.
        7. removeRedundantBlank : Start to remove redundant blanks.
    """

    # Step1: Transfer to lowercase
    if toLowercase == True:
        str = to_lowercase(str)

    # Step2: Convert the full-width characters to be half-width.
    if toHalfwidth == True:
        str = to_halfwidth(str)

    # Step3: Remove URLs
    if removeURL == True:
        str = remove_URL(str)

    # Step4: Decontracted abbreviations.
    if toDecontract == True:
        str = to_decontract(str)

    # Step5: Transfer hexcode to error name.
    if parseHpHexcode == True:
        str = parse_hp_hexcode(str)

    # Step6: Remove special punctuation
    if removePunctuation == True:
        str = remove_punctuation(str, remove_dot=removeDot)

    # Step7: Remove redundant space
    if removeRedundantBlank == True:
        str = remove_redundant_blank(str)
    return str


def process_token(
    str,
    removeStopwords=True,
    removeNonASCII=True,
    lemmatizeVerbs=True,
    stemWords=False,
    replaceIntegerToText=False,
    replaceNumberToBlank=True,
    methodOption=0,
    noReplaceCodeName=True,
):

    """
    Parameters:
        1. removeStopwords: Start to remove stopwords.
        2. removeNonASCII : Start to remove non-ASCII code
        3. lemmatizeVerbs : Start to lemmatize word.
        4. stemWords      : Start to stem word.
        5. replaceIntegerToText : Start to replace integer to text representation.(Just only include positive integers.)
        6. replaceNumberToBlank : Start to replace number to special text representation.
        6-1. methodOption : About choose replaceNumberToBlank method.
                            If set 0, start to replace to be blank.
                            If set 1, Start to replace to be string number.
        6-2. noReplaceCodeName : Contain the special words(HP hexcode name)
    """

    tokens = str.split(" ")

    # Step 1
    if removeStopwords == True:
        tokens = remove_stopwords(tokens)

    # Step2
    if removeNonASCII == True:
        tokens = remove_non_ascii(tokens)

    # Step 3
    if lemmatizeVerbs == True:
        tokens = lemmatize_verbs(tokens)

    # Step 4
    if stemWords == True:
        tokens = stem_words(tokens)

    # Step 5
    if replaceIntegerToText == True:
        tokens = replace_integers2text(tokens)

    # Step 6
    if replaceNumberToBlank == True:
        tokens = replace_number2blank(
            tokens, method_option=methodOption, no_replace_code_name=noReplaceCodeName
        )

    return tokens
