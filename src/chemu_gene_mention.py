import argparse
import logging
import os
import sys

import pandas as pd

from similarity.similarity_evaluator import SimilarityEvaluator


class ChemuGeneMention:

    def load(self, dir):
        data = []
        files = [os.path.join(dir, f) for f in os.listdir(dir)]
        for file in files:
            if not file.endswith(".txt"): continue

            with open(file, "r") as f:
                text = f.read()

            # annotation file
            annotation_file = file.replace(".txt", ".ann")
            sep = '\t'
            with open(annotation_file, "r") as f:
                lines = f.readlines()
                for l in lines[1:]:
                    entity_type = l.split(sep)[1].split(" ")[0]
                    entity = l.split(sep)[2].rstrip("\n")

                    data.append({"text": text, "entity": entity, "entity_type": entity_type})

        df = pd.DataFrame(data=data)
        return df


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traindir",
                        help="The train directory containing train .txt and .ann files ", required=True)
    parser.add_argument("--testdir",
                        help="The test file directory containing train .txt and .ann files", required=True)

    parser.add_argument("--log-level", help="Log level", default="INFO", choices={"INFO", "WARN", "DEBUG", "ERROR"})

    args = parser.parse_args()
    print(args.__dict__)
    # Set up logging
    logging.basicConfig(level=logging.getLevelName(args.log_level), handlers=[logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    return args


def run(traindir, testdir):
    train = ChemuGeneMention().load(traindir)
    test = ChemuGeneMention().load(testdir)

    # Evaluate text similarity, use unique text
    _, result_detail = SimilarityEvaluator().run(pd.DataFrame(data={"text": test["text"].unique()}),
                                                 pd.DataFrame(data={"text": train["text"].unique()}),
                                                 column="text")

    # Evaluate per entity type
    for e in train["entity_type"].unique():
        print("Running entity type {}".format(e))
        query = "entity_type == '{}'".format(e)
        train_entity = train.query(query)
        test_entity = test.query(query)
        _, result_detail = SimilarityEvaluator().run(test_entity, train_entity, column="entity")

    # Evaluate overall entity
    _, result_detail = SimilarityEvaluator().run(test, train, column="entity")


if "__main__" == __name__:
    args = _parse_args()
    run(args.traindir, args.testdir)
