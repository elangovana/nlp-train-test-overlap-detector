[![Build Status](https://travis-ci.org/elangovana/nlp-train-test-overlap-detector.svg?branch=master)](https://travis-ci.org/elangovana/nlp-train-test-overlap-detector)

# NLP Train-Test overlap detector

This repo contains the source code for `Memorization vs. Generalization : Quantifying Data Leakage in NLP Performance Evaluation` EACL 2021 Main conference paper
https://www.aclweb.org/anthology/2021.eacl-main.113/

Please cite paper

```text
@inproceedings{elangovan-etal-2021-memorization,
    title = "Memorization vs. Generalization : Quantifying Data Leakage in {NLP} Performance Evaluation",
    author = "Elangovan, Aparna  and
      He, Jiayuan  and
      Verspoor, Karin",
    booktitle = "Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume",
    month = apr,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.eacl-main.113",
    pages = "1325--1335",
    abstract = "Public datasets are often used to evaluate the efficacy and generalizability of state-of-the-art methods for many tasks in natural language processing (NLP). However, the presence of overlap between the train and test datasets can lead to inflated results, inadvertently evaluating the model{'}s ability to memorize and interpreting it as the ability to generalize. In addition, such data sets may not provide an effective indicator of the performance of these methods in real world scenarios. We identify leakage of training data into test data on several publicly available datasets used to evaluate NLP tasks, including named entity recognition and relation extraction, and study them to assess the impact of that leakage on the model{'}s ability to memorize versus generalize.",
}
```

More details in Notebook [Similarity.ipynb](Similarity.ipynb) and [SimilaritySplitter.ipynb](SimilaritySplitter.ipynb)


## AIMed dataset

### Random split
- This compares the text between train and test using random split

    ```bash
    export PYTHONPATH=./src
    python src/aimed_random.py --trainfile "trainfile.json" 
    
    ```
    
## Biocreative II gene mention

[Biocreative II gene mention](https://biocreative.bioinformatics.udel.edu/tasks/biocreative-ii/) overlap. Please download the test and train files for this task from the BioCreative Website.

- This compares the text between train and test

    ```bash
    export PYTHONPATH=./src

    outputdir=split
    python src/bc2_gene_mention.py --trainfile "tests/data/bc2_gene_mention.in" --testfile "tests/data/bc2_gene_mention.in" --type text --outdir $outputdir --extraeval "predictions.txt,testGENE.eval,ALTGENE.eval"
    
    ```

- This compares the annotation or gene names between train and test

    ```bash
    export PYTHONPATH=./src
  
    outputdir=split
    python src/bc2_gene_mention.py --trainfile "tests/data/bc2_gene_mention.eval" --testfile "tests/data/testGene.eval" --type eval --outdir $outputdir
    
    ```

- Evaluating splits

    Download the eval script from provided as part of training data
    [Biocreative II gene mention](https://biocreative.bioinformatics.udel.edu/tasks/biocreative-ii/) overlap. Please download the test and train files for this task from the BioCreative Website.

     ```bash
     base_dir=tmp
     predictionprefix=result_test_pred.txt
     n=1
     t=0
     perl $base_dir/alt_eval.perl -gene $base_dir/split/testGENE.eval_${n}_${t}.txt -altgene $base_dir/split/ALTGENE.eval_${n}_${t}.txt $base_dir/split/${predictionprefix}_${n}_${t}.txt
     ```


## Biocreative III Protein interaction article classification

[Biocreative III Protein interaction article classification](https://biocreative.bioinformatics.udel.edu/resources/corpora/biocreative-iii-corpus/) overlap. Please download the test and train TSV files for this task from the BioCreative Website.

- This compares the text overlap between train and test

    ```bash
    export PYTHONPATH=./src
    export datadir=./tmp
    python src/bc3_article_classification.py --trainfile $datadir/bc3_act_all_records.tsv --testfile $datadir/bc3_act_all_records_test.tsv --testgoldfile $datadir/bc3_act_gold_standard_test.tsv --predictionsfile $datadir/bc3act-output.csv
    
    ```


## Chemu entity recognition

- This compares the text overlap between train and test

    ```bash
    export PYTHONPATH=./src
    python src/chemu_gene_mention.py --traindir "train" --testdir "test" 
    
    ```
    

## SST 2 datatset

```bash
export PYTHONPATH=./src
export datadir=./tmp
python src/sst2_dataset.py  --trainfile $datadir/train.tsv --testfile $datadir/test.tsv --dictionary $datadir/dictionary.txt  --predictionsfile $datadir/sst2-output.csv --sentiment $datadir/sentiment_labels.txt 

```

