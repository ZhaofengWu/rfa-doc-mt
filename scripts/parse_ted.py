# Adapted from https://github.com/idiap/HAN_NMT/blob/2564826d7653eb5c69c982f569d90adb00a15f04/preprocess_TED_zh-en/get_text.py

import os
import re
import sys


def main(input_dir, output_dir, src, tgt):
    year = 14 if src in {"es", "ru"} or tgt in {"es", "ru"} else 15

    f_s = open(os.path.join(input_dir, f"train.tags.{src}-{tgt}.{src}"))
    f_t = open(os.path.join(input_dir, f"train.tags.{src}-{tgt}.{tgt}"))
    f_s_o = open(os.path.join(output_dir, f"train.raw.{src}"), "w")
    f_t_o = open(os.path.join(output_dir, f"train.raw.{tgt}"), "w")
    f_doc = open(os.path.join(output_dir, f"train.doc_end_indices"), "w")

    count = 0
    is_empty = True
    for ls, lt in zip(f_s, f_t):
        if ls.startswith("<url>"):
            if not lt.startswith("<url>"):
                print("<url> error " + str(count))
                break
            if count > 0 and not is_empty:
                f_doc.write(str(count - 1) + "\n")
                is_empty = True
        elif not ls.startswith("<"):
            if ls.strip() != "" and lt.strip() != "":
                f_s_o.write(ls.strip() + "\n")
                f_t_o.write(lt.strip() + "\n")
                count += 1
                is_empty = False
    if not is_empty:
        f_doc.write(str(count - 1) + "\n")

    f_s.close()
    f_t.close()
    f_s_o.close()
    f_t_o.close()
    f_doc.close()

    count = 0  # accumulated within the test sets, but not across dev/test
    portions = ["dev2010", "tst2010", "tst2011", "tst2012", "tst2013"]
    if year == 14:
        portions = portions[:-1]
    for portion in portions:
        f_s = open(os.path.join(input_dir, f"IWSLT{year}.TED.{portion}.{src}-{tgt}.{src}.xml"))
        f_t = open(os.path.join(input_dir, f"IWSLT{year}.TED.{portion}.{src}-{tgt}.{tgt}.xml"))

        is_dev = portion.startswith("dev")
        split = "valid" if is_dev else "test"

        f_s_o = open(os.path.join(output_dir, f"{split}.raw.{src}"), "a")
        f_t_o = open(os.path.join(output_dir, f"{split}.raw.{tgt}"), "a")
        f_doc = open(os.path.join(output_dir, f"{split}.doc_end_indices"), "a")
        f_s_doc = open(os.path.join(output_dir, f"{split}.metadata.{src}"), "a")
        f_t_doc = open(os.path.join(output_dir, f"{split}.metadata.{tgt}"), "a")

        is_empty = True
        for ls, lt in zip(f_s, f_t):
            if ls.startswith("<talkid>"):
                if not lt.startswith("<talkid>"):
                    print("<talkid> error " + str(count) + " " + portion)
                    break
                s = re.sub("(^\<talkid\>)(.*)(\</talkid\>)", "\g<2>", ls).strip()
                t = re.sub("(^\<talkid\>)(.*)(\</talkid\>)", "\g<2>", lt).strip()

                if s != t:
                    print("error " + str(count) + " " + portion)
                    break

                f_s_doc.write(ls.strip() + "\n")
                f_t_doc.write(lt.strip() + "\n")
                if count > 0 and not is_empty:
                    f_doc.write(str(count - 1) + "\n")

            elif ls.startswith("<seg"):
                if not lt.startswith("<seg"):
                    print("<seg error " + str(count) + " " + portion)
                    break

                ls = re.sub("(^\<seg.*\>)(.*)(\</seg\>)", "\g<2>", ls).strip()
                lt = re.sub("(^\<seg.*\>)(.*)(\</seg\>)", "\g<2>", lt).strip()

                if ls.strip() != "" and lt.strip() != "":
                    f_s_o.write(ls + "\n")
                    f_t_o.write(lt + "\n")
                    count += 1
                    is_empty = False
            else:

                f_s_doc.write(ls.strip() + "\n")
                f_t_doc.write(lt.strip() + "\n")
        if not is_empty:
            f_doc.write(str(count - 1) + "\n")

        if is_dev:
            count = 0

        f_s.close()
        f_t.close()
        f_s_o.close()
        f_t_o.close()
        f_doc.close()
        f_s_doc.close()
        f_t_doc.close()


if __name__ == "__main__":
    main(*sys.argv[1:])
