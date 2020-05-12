import pandas as pd
import math,re
from collections import Counter


WORD = re.compile(r'\w+')

#calcola la cos similarity di due vettori
def get_cosine(text1,text2):
    vec1 = text_to_vector(text1)
    vec2 = text_to_vector(text2)
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator

def text_to_vector(text):
     words = WORD.findall(text)
     return Counter(words)


def findPossibleMatchings(record,source,min_similarity):
    record2text = " ".join([val for k,val in record.to_dict().items() if k not in ['id']])
    source_without_id = source.copy()
    source_without_id = source_without_id.drop(['id'],axis=1)
    source_ids = source.id.values
    #for a faster iteration
    source_without_id = source_without_id.values
    possibleMatchings = []
    for idx,row in enumerate(source_without_id):
        currentRecord = " ".join(row)
        currentSimilarity = get_cosine(record2text,currentRecord)
        if currentSimilarity>=min_similarity:
            possibleMatchings.append((record['id'],source_ids[idx]))
    return pd.DataFrame(possibleMatchings,columns=['ltable_id','rtable_id'])


