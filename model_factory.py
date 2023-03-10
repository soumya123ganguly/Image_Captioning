from rnn import CaptionsRNN
from lstm1 import CaptionsLSTM1
from lstm2 import CaptionsLSTM2

# Build and return the model here based on the configuration.
def get_model(config_data, vocab):
    hidden_size = config_data['model']['hidden_size']
    embedding_size = config_data['model']['embedding_size']
    model_type = config_data['model']['model_type']

    # You may add more parameters if you want
    if model_type == 'LSTM1':
        print(1, model_type)
        return CaptionsLSTM1(hidden_size, embedding_size, vocab, config_data['generation'])
    elif model_type == 'RNN':
        print(2, model_type)
        return CaptionsRNN(hidden_size, embedding_size, vocab, config_data['generation'])
    elif model_type == 'LSTM2':
        print(3, model_type)
        return CaptionsLSTM2(hidden_size, embedding_size, vocab, config_data['generation'])
    else:
        raise Exception("Model not implemented")