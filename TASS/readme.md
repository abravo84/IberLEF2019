###### About the task

The workshop and shared task "Sentiment Analysis at SEPLN (TASS)" has been held since 2012, under the umbrella of the International Conference of the Spanish Society for Natural Language Processing (SEPLN). TASS was the first shared task on sentiment analysis in Twitter in Spanish. Spanish is the second language used in Facebook and Twitter [1], which calls for the development and availability of language-specific methods and resources for sentiment analysis. The initial aim of TASS was the furtherance of research on sentiment analysis in Spanish with a special interest on the language used in Twitter.

Although sentiment analysis is still an open problem, the Organization Committee would like to foster research on other tasks related to the processing of the semantics of texts written in Spanish. Consequently, the name of the workshop/shared task has been changed to "Workshop on Semantic Analysis at SEPLN (TASS)".

The Organization Committee appeals to the research community to propose and organize evaluation tasks related to other semantic tasks in the Spanish language. New tasks provide an opportunity to create linguistic resources, evaluate their usefulness, and promotes the consolidation of a community of researchers interested in the addressed topics. Thus, we encourage the semantic processing community to propose and submit an evaluation tasks (see Proposal of Tasks).

###### Task: Sentiment Analysis at Tweet level

The tasks we propose are the natural evolution from TASS 2018 Task 1. The first aim of this task is the furtherance of research on sentiment analysis in Spanish with a special interest on the language used in Twitter. The target community for this task is any research group working in this area. Traditionally, we have had about ten systems each year. The task tries to attract Hispanic American groups and offering a common meeting point in the research of this type of task.

This task focuses on the evaluation of polarity classification systems of tweets written in Spanish. The submitted systems will have to face up with the following challenges:

    Lack of context: Remember, tweets are short (up to 240 characters).
    Informal language: Misspellings, emojis, onomatopeias are common.
    (Local) multilinguality: The training, tests and development corpus contains tweets written in the Spanish language spoken in Spain, Peru and Costa Rica.
    Generalization: The systems will be assessed with several corpora, one is the test set of the training data, so it follows a similar distribution; the second corpus is the test set of the General Corpus of TASS (see previous editions), which was compiled some years ago, so it may be lexical and semantic different from the training data. Furthermore, the system will be evaluated with test sets of tweets written in the Spanish language spoken in different American countries.

The participants will be provided with a training, a development and several test corpora (see important dates). All the corpora are annotated with 4 different levels of opinion intensity (P, N, NEU, NONE).

In case the participants submit a supervised or semi-supervised system, it must be only trained with provided training data and it is totally forbidden the use of other training set. However, linguistic resources like lexicons, vectors of word embeddings or knowledge bases can be used. We want a fair competition and furtherance the creativity, so we want to assess the originality of the systems given the same set of training data.

###### Subtasks

    Subtask-1: Monolingual Sentiment Analysis: Training and test using each InterTASS dataset (ES-Spain, PE-Peru, CR-Costa Rica and UR-Uruguay).
    Subtask-2: Cross-lingual. Training a selection of any dataset and use a different one to test, in order to test the dependency of systems on a language.
