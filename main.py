import bert_classification as bc


while(True):
    review_text = input()

    if review_text == 'break':
        break

    print(bc.classify(review_text))


