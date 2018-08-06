from _hangul import normalize

str = "한글테스트 1234567890 !@#$%%^&*() ,.<>?[];'~+` ABCDEFG abcdefg"
print("Original: ", str)
print("english=True, number=False, punctuation=False : ", normalize(str, english=True, number=False, punctuation=False))
print("english=False, number=True, punctuation=False : ", normalize(str, english=False, number=True, punctuation=False))
print("english=False, number=False, punctuation=True : ", normalize(str, english=False, number=False, punctuation=True))