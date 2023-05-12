from transformers import BertTokenizer, AutoModelForSequenceClassification
import pickle


def load_model(predict_type):
    path = f'aipfanatic/aip-bert-{predict_type}'
    model = AutoModelForSequenceClassification.from_pretrained(path) #<class 'transformers.models.bert.modeling_bert.BertModel'>
    tokenizer = BertTokenizer.from_pretrained(path)

    dict_path = f'./dicts/dict_{predict_type}.pkl'
    with open(dict_path, 'rb') as file:
        loaded_dict = pickle.load(file)

    return (model, tokenizer, loaded_dict, predict_type)