


if __name__ == '__main__':
    input_path = "/home/upf/Downloads/TT040619.008.csv"
    sum = 0
    for line in open(input_path):
        if not "DAVID MORENO" in line:
            continue

        fields = line.strip().split("\t")
        if len(fields) < 3:
            continue



        if fields[5] != "":
            print("\t".join([fields[3], fields[4], fields[5], fields[12],fields[13], fields[14]]))



        sum+= float(fields[5])

    print(sum)