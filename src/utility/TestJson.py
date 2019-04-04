import json

count = 0


#def getNodeCombination():
#    for i in range

#adv "LeakyReLU" "PReLU" "ELU" "ThresholdedReLU" "ReLU"
def CreateJsonTextFiles ():
    activation = ["softmax","relu","tanh","sigmoid","hard_sigmoid","exponential","linear","elu","selu","softplus","softsign"]
    optimizer = ["SGD", "RMSprop", "Adagrad", "Adadelta", "Adam", "Adamax", "Nadam"]
    loss = ["mean_squared_error", "mean_absolute_error", "mean_absolute_percentage_error", "mean_squared_logarithmic_error", "squared_hinge", "hinge", "categorical_hinge", "logcosh", "categorical_crossentropy", "sparse_categorical_crossentropy", "binary_crossentropy", "kullback_leibler_divergence", "poisson", "cosine_proximity"]
    print ("Activations: ", len(activation))
    print ("optimizers: ", len(optimizer))
    print ("losses: ", len(loss))
    hiddenlayercountmin = 1
    hiddenlayercountmax = 10
    layernodemin = 2
    layernodemax = 250
    count = 0
    data = {}
    data['models'] = []

    nodecombination = []



    for layercount in range (hiddenlayercountmin, hiddenlayercountmax+1):
        model = {}
        model['layer'] = []
        for layer_iter in range (0, layercount):
            layer = {}
            for nodecoutn in range (layernodemin, layernodemax+1):
                nodes = {}
                layer['nodes'] = nodecoutn
                layer['activation'] = "relu"
            model['layer'].append(layer)
        data['models'].append(model)

    with open('json.json', 'w') as outfile:
        json.dump(data, outfile)