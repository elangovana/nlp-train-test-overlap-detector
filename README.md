[![Build Status](https://travis-ci.org/elangovana/nlp-train-test-overlap-detector.svg?branch=master)](https://travis-ci.org/elangovana/nlp-train-test-overlap-detector)

# NLP Train-Test overlap detector

## Biocreative II gene mention

[Biocreative II gene mention](https://biocreative.bioinformatics.udel.edu/tasks/biocreative-ii/) overlap. Please download the test and train files for this task from the BioCreative Website.

- This compares the text between train and test

    ```bash
    export PYTHONPATH=./src
    python src/bc2_gene_mention.py --trainfile "tests/data/bc2_gene_mention.in" --testfile "tests/data/bc2_gene_mention.in" --type text
    
    ```

- This compares the annotation or gene names between train and test

    ```bash
    export PYTHONPATH=./src
    python src/bc2_gene_mention.py --trainfile "tests/data/bc2_gene_mention.eval" --testfile "tests/data/bc2_gene_mention.eval" --type eval
    
    ```

## Biocreative III Protein interaction article classification

[Biocreative III Protein interaction article classification](https://biocreative.bioinformatics.udel.edu/resources/corpora/biocreative-iii-corpus/) overlap. Please download the test and train TSV files for this task from the BioCreative Website.

- This compares the text overlap between train and test

    ```bash
    export PYTHONPATH=./src
    python src/bc3_article_classification.py --trainfile "bc3_act_all_records.tsv" --testfile "bc3_act_all_records_test.tsv" 
    
    ```
