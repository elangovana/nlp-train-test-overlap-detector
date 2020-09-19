[![Build Status](https://travis-ci.org/elangovana/nlp-train-test-overlap-detector.svg?branch=master)](https://travis-ci.org/elangovana/nlp-train-test-overlap-detector)

# NLP Train-Test overlap detector

1. [Biocreative II gene mention](https://biocreative.bioinformatics.udel.edu/tasks/biocreative-ii/) overlap. Please download the test and train files for this task from the BioCreative Website.

- This compares the text between train and test

    ```bash
    export PYTHONPATH=./src
    python src/biocreative_gene_mention.py --trainfile "biocreative_gene_mention.in" --testfile "biocreative_gene_mention.in" --type text
    
    ```

- This compares the annotation or gene names between train and test

    ```bash
    export PYTHONPATH=./src
    python src/biocreative_gene_mention.py --trainfile "biocreative_gene_mention.eval" --testfile "biocreative_gene_mention.eval" --type eval
    
    ```