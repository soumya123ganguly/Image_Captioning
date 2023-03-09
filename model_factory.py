from lstmcap import LSTMCap

# Build and return the model here based on the configuration.
def get_model(config_data, vocab):
    hidden_size = config_data['model']['hidden_size']
    embedding_size = config_data['model']['embedding_size']
    model_type = config_data['model']['model_type']

    # You may add more parameters if you want
    return LSTMCap(vocab)
    #return CaptionsLSTM1(hidden_size, embedding_size, vocab, config_data['generation'])
