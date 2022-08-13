import re
#print(re.search('(.*?)', '"\ufeff""86680728811_272953252761568"""').group(1))

string = '"\ufeff""86680728811_272953252761568"""'
print(string.replace('"\ufeff""', '').replace('"""',''))