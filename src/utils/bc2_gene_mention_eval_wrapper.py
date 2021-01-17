# *****************************************************************************
# * Copyright 2020 Amazon.com, Inc. and its affiliates. All Rights Reserved.  *
#                                                                             *
# Licensed under the Amazon Software License (the "License").                 *
#  You may not use this file except in compliance with the License.           *
# A copy of the License is located at                                         *
#                                                                             *
#  http://aws.amazon.com/asl/                                                 *
#                                                                             *
#  or in the "license" file accompanying this file. This file is distributed  *
#  on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either  *
#  express or implied. See the License for the specific language governing    *
#  permissions and limitations under the License.                             *
# *****************************************************************************
import argparse
import logging
import os
import re
import subprocess
import sys

F_SCORE = "F-score"

RECALL = "Recall"

PRECISION = "Precision"

FN = "FN"

TP = "TP"

FP = "FP"


class BC2GeneMentionEvalWrapper:
    """
    Provides a python wrapper over the original Perl BC2 gene mention eval script
    """

    def __init__(self, script_path=None):
        self.script_path = script_path or os.path.join(os.path.dirname(__file__), "alt_eval.perl")

    def get_score(self, genefile, altgenefile, predictionsfile):
        cmd = [
            "perl"
            , self.script_path
            , "-gene"
            , genefile
            , "-altgene"
            , altgenefile
            , predictionsfile

        ]

        response = self._run_shell(cmd)

        predictions, score = self._parse_eval_response(response)
        return predictions, score

    @property
    def _logger(self):
        return logging.getLogger(__name__)

    def _parse_eval_response(self, eval_response):
        self._logger.debug(eval_response)

        try:
            eval_response_parts = eval_response.split("\n\n")
            raw_predictions = eval_response_parts[0]
            predictions = self._parse_raw_predictions(raw_predictions)

            raw_scores = eval_response_parts[1]
            scores_dict = self._parse_raw_eval_scores(raw_scores)

        except Exception as e:
            self._logger.info(eval_response)
            raise e
        return predictions, scores_dict

    def _parse_raw_predictions(self, raw_predictions):
        """
        The raw_eval_scores looks like this
            FP|BC2GM075521296|11 13|ATP
            FP|BC2GM093464345|82 85|Pax -
            FP|BC2GM093464345|86 88|QNR
            FP|BC2GM093464345|89 98|P0 promoter
            FP|BC2GM093464345|105 116|heterologous

        :param raw_predictions:
        :return:
        """
        result = []
        for line in raw_predictions.split("\n"):
            line_parts = line.split("|")
            type = line_parts[0]
            assert type.lstrip("*") in (
                "FP", "TP", "TN", "FN"), 'Expected {} to be in (FP, TP, TN, FN), {}'.format(line[0], line)

            docid = line_parts[1]
            start_end = line_parts[2]
            entity_name = line_parts[3]
            alt_gene = None
            alt_gene_start_end = None

            if type.lstrip("*") == "TP":
                start_end = line_parts[3]
                entity_name = line_parts[2]
                alt_gene = line_parts[4]
                alt_gene_start_end = line_parts[5]

            result.append({
                "type": type,
                "docid": docid,
                "start_end": start_end,
                "entity_name": entity_name,
                "alt_gene": alt_gene,
                "alt_gene_start_end": alt_gene_start_end,
            })
        return result

    def _parse_raw_eval_scores(self, raw_eval_scores):
        """
        Expect score to look something like this
            TP: 5471
            FP: 1590
            FN: 860
            Precision: 0.774819430675542 Recall: 0.864160480176907 F: 0.81705495818
        :param eval_scores:
        :return: a dictionary of scores
        """
        result = {}

        result[FP] = int(self._extract_pattern(r"FP: (\d+)\n", raw_eval_scores))

        result[TP] = int(self._extract_pattern(r"TP: (\d+)\n", raw_eval_scores))

        result[FN] = int(self._extract_pattern(r"FN: (\d+)\n", raw_eval_scores))

        result[PRECISION] = float(self._extract_pattern(r"Precision: (0?\.?\d+)", raw_eval_scores))
        result[RECALL] = float(self._extract_pattern(r"Recall: (0?\.?\d+)", raw_eval_scores))
        result[F_SCORE] = float(self._extract_pattern(r"F: (0?\.?\d+)", raw_eval_scores))

        return result

    def _extract_pattern(self, precision_pattern, payload):
        matches = re.search(precision_pattern, payload)
        if not matches: raise Exception(
            "Unable to parse pattern in {} response \n{}".format(precision_pattern, payload))
        return matches.groups()[0]

    def _run_shell(self, cmd):
        """
        Runs a shell command
        :param cmd: The cmd to run
        """
        self._logger.info("Running command\n{}".format(" ".join(cmd)))

        out = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)
        stdout, stderr = out.communicate()
        result = stdout.decode(encoding='utf-8')
        if stderr:
            error_msg = stderr.decode(encoding='utf-8')
            print(error_msg)
            raise Exception(error_msg)

        return result


def run_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--genefile",
                        help="The gene  file ", required=True)
    parser.add_argument("--altgenefile",
                        help="The altgene file ", required=True)
    parser.add_argument("--predictionsfile",
                        help="The predictions file ", required=True)

    parser.add_argument("--log-level", help="Log level", default="INFO", choices={"INFO", "WARN", "DEBUG", "ERROR"})

    args = parser.parse_args()
    print(args.__dict__)
    # Set up logging
    logging.basicConfig(level=logging.getLevelName(args.log_level), handlers=[logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    predictions, score = BC2GeneMentionEvalWrapper().get_score(args.genefile, args.altgenefile, args.predictionsfile)
    print(predictions)
    print(score)


if __name__ == '__main__':
    run_main()
