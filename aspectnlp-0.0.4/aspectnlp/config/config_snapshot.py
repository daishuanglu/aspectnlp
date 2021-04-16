# configuration for initial training models and network settings
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
    pretrained=''
    transfer_domain = []
    model_name='snapshot'
    sentence=None
    target=None
    transfer_data =None

    test=True
    verbose=1
    trials=1


good_lap_attributes = ['battery#operation_performance', 'battery#quality', 'company#general', 'cpu#operation_performance', 'display#design_features', 'display#general', 'display#operation_performance', 'display#quality', 'display#usability', 'graphics#general', 'graphics#quality', 'hard_disc#design_features', 'hard_disc#quality', 'keyboard#design_features', 'keyboard#general', 'keyboard#operation_performance', 'keyboard#quality', 'keyboard#usability', 'laptop#connectivity', 'laptop#design_features', 'laptop#general', 'laptop#miscellaneous', 'laptop#operation_performance', 'laptop#portability', 'laptop#price', 'laptop#quality', 'laptop#usability', 'memory#design_features', 'motherboard#quality', 'mouse#design_features', 'mouse#general', 'mouse#operation_performance', 'mouse#quality', 'mouse#usability', 'multimedia_devices#general', 'multimedia_devices#operation_performance', 'multimedia_devices#quality', 'optical_drives#quality', 'os#general', 'os#operation_performance', 'os#usability', 'power_supply#quality', 'shipping#quality', 'software#design_features', 'software#general', 'software#miscellaneous', 'software#operation_performance', 'software#usability', 'support#price', 'support#quality']

