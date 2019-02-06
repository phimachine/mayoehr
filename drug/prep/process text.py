codes=[]

with open('data/icd10cm_codes_2019.txt','r') as file:
    for line in file:
        words=line.split()
        codes.append(words[0])

with open("data/all_icd10.txt","w") as file:
    file.write("code\n")
    for code in codes:
        file.write(code+"\n")

