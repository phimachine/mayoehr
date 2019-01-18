with open("drugs.csv","w") as file:
    for i in range(36,51):
        file.write("^T"+str(i)+"\n")