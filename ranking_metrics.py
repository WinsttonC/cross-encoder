
def mrr_with_logs(data):
    mrr_metrics = []

    log_question_counter = 0
    log_checker_1 = 0
    log_checker_2 = 0
    log_checker_3 = 0

    questions = data['question'].to_list()
    for question in questions:
        relevant_docs = get_rel_chunks(question)

        pos = 1
        for doc in relevant_docs:
            a = data.loc[data['question']==question, 'answer'].values[0]
            if a in doc:
                mrr_metrics.append(1/pos)
                # print(f'CHECK1-полное совпадение подстроки в чанке\n\nПравильный ответ\n\n{a}\n\nЧанк:\n\n{doc}')
                log_checker_1 += 1
                break
            elif longest_common_substring(a, doc) >= len(a.split(" ")) // 4:
                mrr_metrics.append(1/pos)
                print(f'CHECK2-пересечение\n\nПравильный ответ: {a}\nЧанк:\n {doc}\n\nПересечение: {longest_common_substring(a, doc)}\n\n')
                log_checker_2 += 1
                break
            else:
                # print(f'CHECK3-нет пересечения и подстроки\n\nОтвет:\n\n{a}\n\nЧанк:\n\n {doc}')
                log_checker_3 += 1
            pos += 1
        
        log_question_counter += 1

    print(f'Q:{log_question_counter}')
    print(f'C1:{log_checker_1}')
    print(f'C2:{log_checker_2}')
    print(f'C3:{log_checker_3}')

    while len(mrr_metrics) < len(questions):
        mrr_metrics.append(0)
    return f'Mean MRR = {sum(mrr_metrics)/len(mrr_metrics)}'