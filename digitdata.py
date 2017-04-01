def readDigitData():

    # Atributo de clasificación
    # -------------------------
    
    digitos_atributo_clasificacion='digito'
    
    
    # Valores de clasificación:
    # -------------------------
    
    digitos_clases=['0','1','2','3','4','5','6','7','8','9']
    
    
    # Atributos (o características):
    # ------------------------------
    
    digitos_atributos=["char" + str(x) for x in range(28*28)]
    
    
    # Valores posibles de cada atributo:
    # ----------------------------------
    
    digitos_valores_atributos={"char" + str(x): ['+', '-'] for x in range(28*28)}
    
    
    # Ejemplos del conjunto de entrenamiento:
    # ---------------------------------------
    
    digitos_entr=[]
    
    f = open('digitdata\\trainingimages', 'r')
    char_count = 0
    for line in f:
        if char_count % (28*28) == 0:
            digitos_entr.append([])
        for c in range(28):
            if line[c] == '+' or line[c] == '#':
                digitos_entr[int(char_count/(28*28))].append('+')
            else:
                digitos_entr[int(char_count/(28*28))].append('-')
            char_count += 1
    
    
    # Clasificación de los ejemplos del conjunto de entrenamiemto:
    # ------------------------------------------------------------
    
    # En el mismo orden en el que aparecen los ejemplos
    
    digitos_entr_clas=[]
    
    f = open('digitdata\\traininglabels', 'r')
    for line in f:
        digitos_entr_clas.append(line[0])
    
    
    # Ejemplos del conjunto de validación:
    # ------------------------------------
    
    digitos_valid=[]
    
    f = open('digitdata\\validationimages', 'r')
    char_count = 0
    for line in f:
        if char_count % (28*28) == 0:
            digitos_valid.append([])
        for c in range(28):
            if line[c] == '+' or line[c] == '#':
                digitos_valid[int(char_count/(28*28))].append('+')
            else:
                digitos_valid[int(char_count/(28*28))].append('-')
            char_count += 1
    
    
    # Clasificación de los ejemplos del conjunto de validación:
    # =========================================================
    
    votos_valid_clas=[]
    
    f = open('digitdata\\validationlabels', 'r')
    for line in f:
        votos_valid_clas.append(line[0])
    
    
    # Ejemplos del conjunto de prueba (o test):
    # =========================================
    
    digitos_test=[]
    
    f = open('digitdata\\testimages', 'r')
    char_count = 0
    for line in f:
        if char_count % (28*28) == 0:
            digitos_test.append([])
        for c in range(28):
            if line[c] == '+' or line[c] == '#':
                digitos_test[int(char_count/(28*28))].append('+')
            else:
                digitos_test[int(char_count/(28*28))].append('-')
            char_count += 1
                
                      
                     
    # Clasificación de los ejemplos del conjunto de prueba (o test):
    # ==============================================================
                                  
    digitos_test_clas=[]
    
    f = open('digitdata\\testlabels', 'r')
    for line in f:
        digitos_test_clas.append(line[0])

    return (digitos_atributo_clasificacion, digitos_clases, digitos_atributos,
            digitos_valores_atributos, digitos_entr, digitos_entr_clas,
            digitos_valid, votos_valid_clas, digitos_test, digitos_test_clas)
