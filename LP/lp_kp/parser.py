import string
input_file = open("FamilyTree.ged", 'r')
output_file = open("result.pl", 'w')
husb_id = ""
wife_id = ""
name = ""
surname = ""
sex = ""
n = 1
table = dict() 
for input_string in input_file:
    if input_string == "\n":
        continue
    elif input_string.find("GIVN", 0, len(input_string)) > -1:
        name = input_string[7:len(input_string) - 1]

    elif input_string.find("SURN", 0, len(input_string)) > -1:
        surname = input_string[7:len(input_string) - 1]
        table[n] = name + ' ' + surname
        n = n + 1
        
    elif input_string.find("HUSB", 0, len(input_string)) > -1:
        i = input_string.index("I", 0, len(input_string))
        husb_id = input_string[i+2:len(input_string) - 2]

    elif input_string.find("WIFE", 0, len(input_string)) > -1:
        i = input_string.index("@", 0, len(input_string))
        wife_id = input_string[i+3:len(input_string) - 2] 

    elif input_string.find("CHIL", 0, len(input_string)) > -1:
        i = input_string.index("@", 0, len(input_string))
        child_id = input_string[i+3:len(input_string) - 2] 
        output_file.write("father('{0}', '{1}').\n".format(table[int(husb_id)], table[int(child_id)]))
        output_file.write("mother('{0}', '{1}').\n".format(table[int(wife_id)], table[int(child_id)]))

input_file.close()
output_file.close()
table.clear()
