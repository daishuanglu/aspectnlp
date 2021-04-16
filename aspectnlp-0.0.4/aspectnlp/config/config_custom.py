# set transfer data path
class transfer_data_format():
    train='../custom_data/data_train.csv'
    test='../custom_data/data_test.csv'
    asp_header='aspect'
    txt_header='Response'
    senti_header='sentiment'
    columns=['RowKey','Question','Response','aspect','sentiment','keyword']
    senti_code={-1:'negative', 0:'neutral',1:'positive'}

# configuration for transfer training models and network settings
class configuration():
    #learning
    lr=0.001
    l2=0
    monmentum=0.99
    epochs=30
    batch_size=32
    grad_clip=5
    lr_decay=0.0

    # logging
    log_interval=10
    test_interval=100
    save_interval=1000
    save_dir="./backup"

    # data
    shuffle=True
    embed_dim=300
    aspect_embed_dim=300
    unif=0.25
    embed_file="glove"
    aspect_file=""
    years="14_15_16"
    aspects=None
    atsa=False
    r_l='r'
    use_attribute=False
    aspect_phrase=False

    # model CNNs
    model='CNN_Gate_Aspect'
    dropout=0.5
    max_norm=3.0
    kernel_num=100
    kernel_sizes='3,4,5'
    att_dsz=100
    att_method='concat'

    ## CNN_CNN
    lambda_sm=1.0
    lambda_as=1.0

    ## LSTM
    lstm_dsz=300
    lstm_bidirectional=True
    lstm_nlayers=1

    # device
    device=-1
    cuda=False

    # option
    # the pretrained model to load for transfer learning
    pretrained='models/custom_final_steps11600.pt'
    transfer_domain=['models/custom_text.pt','models/custom_as.pt','models/custom_sm.pt']
    # the new model name
    model_name='custom'
    sentence=None
    target=None
    transfer_data=transfer_data_format()
    #transfer_data =None

    test=True
    verbose=1
    trials=1

