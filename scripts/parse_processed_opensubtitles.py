import glob
import os
import sys


def main(input_path, output_path):
    for file in glob.glob(os.path.join(input_path, "*")):
        with open(file) as f:
            lang, split = os.path.basename(file).split("_")
            if split == "dev":
                split = "valid"
            with open(os.path.join(output_path, f"{split}.raw.{lang}"), "w") as o:
                with open(os.path.join(output_path, f"{split}.doc_end_indices"), "w") as dei:
                    for i, line in enumerate(f):
                        sents = line.strip().split(" _eos ")
                        assert len(sents) == 4
                        for line in sents:
                            o.write(f"{line}\n")
                        dei.write(f"{i * 4 + 3}\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
