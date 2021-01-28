import pytagcloud
import random
from konlpy.tag import Okt
from collections import Counter
import os

reviews = []
for file in os.listdir("./result/네이버"):
    if file.endswith(".txt"):
        reviews.append(os.path.join("./result/네이버", file))
for file in os.listdir("./result/다음"):
    if file.endswith(".txt"):
        reviews.append(os.path.join("./result/다음", file))
review_text = []
for review in reviews:
    file = open(review, 'r', encoding='utf-8')
    texts = file.readlines()
    for text in texts:
        review_text.append(text)
    file.close()
print(len(review_text))


ok_twitter = Okt()
nouns = []
tags = []
clean_words = []
for i, review in enumerate(review_text):
    okt = Okt()
    for word in okt.pos(review, stem=True): #어간 추출
        if word[1] in ['Adjective']: #형용사만 추출
            #의미없는 단어를 제외한 단어만 리스트에 포함
            if word[0] not in ['좋다','많다','있다','없다','아니다','굳다','안되다','같다','이다']: #
                clean_words.append(word[0])

count = Counter(clean_words)  # 각 단어 별 빈도계산'''

for w, c in count.most_common(40):  # w : 단어, c : 빈도
    tags.append({'color': (random.randint(0, 255),
                           random.randint(0, 255),
                           random.randint(0, 255)),
                 'tag': w,
                 'size': int(0.3*c)})
print(tags)
tags[0]['size'] = int(0.2*tags[0]['size'])
tags[1]['size'] = int(0.2*tags[1]['size'])
pytagcloud.create_tag_image(tags, '12345678.png', fontname='myfont', size=(1280, 720))
