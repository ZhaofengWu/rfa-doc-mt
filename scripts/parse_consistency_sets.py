import glob
import os
import sys


def main(input_path, output_path):
    for file in glob.glob(os.path.join(input_path, "*.*")):
        with open(file) as f:
            with open(f"{os.path.join(output_path, os.path.basename(file))}.raw", "w") as o:
                for line in f:
                    sents = line.strip().split(" _eos ")
                    assert len(sents) == 4
                    for line in sents:
                        o.write(f"{line}\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
