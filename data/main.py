import json

if __name__ == '__main__':
    # 读取entity-dict字典
    entity_dict_wiki = json.load(open('./entity-dict-wiki.json', 'r'))
    print(entity_dict_wiki.keys())

    # 获取train.json中的所有subject和object的类型
    data = json.load(open('./tacred/origin/test.json', 'r'))
    entity_dict_new = {}
    for item in data:
        subj_type = item['subj_type']
        obj_type = item['obj_type']
        if subj_type not in entity_dict_new.keys():
            entity_dict_new[subj_type] = set()
        if obj_type not in entity_dict_new.keys():
            entity_dict_new[obj_type] = set()

        # 将entity-dict中的实体按照类型放入entity_dict_new中
        entity_dict_new[subj_type].add(' '.join(item['token'][item['subj_start']:item['subj_end'] + 1]))
        entity_dict_new[obj_type].add(' '.join(item['token'][item['obj_start']:item['obj_end'] + 1]))

    for entity_type in entity_dict_new.keys():
        if entity_type in entity_dict_wiki.keys():
            entity_dict_new[entity_type].update(entity_dict_wiki[entity_type])
        entity_dict_new[entity_type] = list(entity_dict_new[entity_type])

    with open('./entity-dict.json', 'w') as f:
        json.dump(entity_dict_new, f)
