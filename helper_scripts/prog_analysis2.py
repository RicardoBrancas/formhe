import glob
import re

with open("analysis/prog_analysis2.csv", "w") as csv_f:
    csv_f.write("instance,i,max,relax,type\n")
    for filename in glob.glob("analysis/data/1010/**/*.log", recursive=True):
        with open(filename) as f:
            i = 0
            for line in f.readlines():
                if not re.match(r'.\d+', line):
                    continue
                match line[0]:
                    case '-':
                        t = "Wrong"
                    case '+':
                        t = "Right"
                    case '!':
                        t = "Error"
                    case '?':
                        t = "Unknown"
                l = line[1:]
                csv_f.write(f'{filename},{i},{len(l.strip())},{l.strip().index("1") + 1},{t}\n')
                i += 1
