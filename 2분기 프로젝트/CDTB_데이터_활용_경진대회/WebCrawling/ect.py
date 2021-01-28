books = {1:'a', 2:'b', 3:'c', 4:'d', 5:'d'}
b_list = {}
while True:
    name = input("회원의 이름을 입력하세요 : ")
    if name == 'q':
        break
    if name not in b_list.keys():
        b_list[name] = []
        print(b_list)
    while True:
        code = input("빌릴 책의 코드 : ")
        if code == 'q':
            break
        code = int(code)
        available = True
        for temp_name in b_list.keys():
            if books[code] in b_list[temp_name]:
                available = False
                break
        if available :
            b_list[name].append(books[code])
        else:
            print("이미 대출된 책입니다.")
        if len(b_list[name]) >= 3:
            break
print("현재 대출자 목록")
print(b_list)
