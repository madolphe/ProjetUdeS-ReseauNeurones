import numpy as np


class TwoLayerClassifier(object):
    def __init__(self, x_train, y_train, x_val, y_val, num_features, num_hidden_neurons, num_classes, activation='relu'):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

        self.num_features = num_features
        self.num_classes = num_classes

        self.net = TwoLayerNet(num_features, num_hidden_neurons, num_classes, activation)

        self.momentum_cache_v_prev = {}

    def train(self, num_epochs=1, lr=1e-3, l2_reg=1e-4, lr_decay=1.0, momentum=0.0):
        """
        Train the model with a cross-entropy loss
        Naive implementation (with loop)

        Inputs:
        - num_epochs: the number of training epochs
        - lr: learning rate
        - l2_reg: the l2 regularization strength
        - lr_decay: learning rate decay.  Typically a value between 0 and 1
        - momentum : variable mu for momentum update

        Returns a tuple for:
        - training accuracy for each epoch
        - training loss for each epoch
        - validation accuracy for each epoch
        - validation loss for each epoch
        """
        loss_train_curve = []
        loss_val_curve = []
        accu_train_curve = []
        accu_val_curve = []

        self.net.reinit()
        self.net.l2_reg = l2_reg

        self.momentum_cache_v_prev = {id(x): np.zeros_like(x) for x in self.net.parameters}  # variable v of momentum

        sample_idx = 0
        num_iter = num_epochs * len(self.x_train)
        for i in range(num_iter):
            # Take a sample
            x_sample = self.x_train[sample_idx]
            y_sample = self.y_train[sample_idx]

            # Forward + Backward
            loss_train = self.net.forward_backward(x_sample, y_sample)

            # Take gradient step
            for w, dw in zip(self.net.parameters, self.net.gradients):
                self.momentum_update(w, dw, lr, momentum)

            # Advance in data
            sample_idx += 1
            if sample_idx >= len(self.x_train):  # End of epoch

                accu_train, loss_train = self.global_accuracy_and_cross_entropy_loss(self.x_train, self.y_train)
                accu_val, loss_val, = self.global_accuracy_and_cross_entropy_loss(self.x_val, self.y_val)

                loss_train_curve.append(loss_train)
                loss_val_curve.append(loss_val)
                accu_train_curve.append(accu_train)
                accu_val_curve.append(accu_val)

                sample_idx = 0

                lr *= lr_decay

        return loss_train_curve, loss_val_curve, accu_train_curve, accu_val_curve

    def predict(self, x):
        """
        Returns the most likely class label associated to x
        Naive implementation (with loop)
        Inputs:
        - X: A numpy array of shape (D, N) containing one or many samples.
        Returns a tuple of:
        - A numpy array of shape (N) containing one or many class label
        """
        if len(x.shape) == 1:  # Predict on one sample
            #############################################################################
            # return the most probable class label for one sample.                      #
            #############################################################################
            return np.argmax(np.exp(self.net.forward(x))/np.sum(np.exp(self.net.forward(x))))
            #############################################################################
            #                          END OF YOUR CODE                                 #
            #############################################################################

        elif len(x.shape) == 2:  # Predict on multiple samples
            #############################################################################
            # return the most probable class label for many samples                     #
            #############################################################################
            pred = np.array([np.exp(self.net.forward(elt)) / np.sum(np.exp(self.net.forward(elt))) for elt in x])
            pred_to_label = np.argmax(pred, axis=1)
            return pred_to_label
            #############################################################################
            #                          END OF YOUR CODE                                 #
            #############################################################################

    def global_accuracy_and_cross_entropy_loss(self, x, y, l2_r=-1.0):
        """
        Compute average accuracy and cross_entropy for a series of N data points.
        Naive implementation (with loop)
        Inputs:
        - x: A numpy array of shape (D, N) containing several samples.
        - y: A numpy array of shape (N) labels as an integer
        - reg: (float) L2 regularization strength
        Returns a tuple of:
        - average accuracy as single float
        - average loss as single float
        """
        if l2_r > 0:
            self.net.l2_reg = l2_r
        #############################################################################
        #       Compute the softmax loss & accuracy for a series of samples X,y .   #
        #############################################################################
        # forward pass:
        softmax_outputs = []
        losses = []
        for i in range(len(x)):
            # On récupère la perte pour notre donnée i courrante:
            losses.append(self.net.forward_backward(x[i], y[i]))
            # On récupère la sortie afin de calculer par la suite la précision du modèle
            # Note: on aurait pu modifier la signature de net.forward_backward afin de
            # récupérer aussi la sortie du réseau
            logit = np.exp(self.net.layer2.last_activ)
            softmax_outputs.append(logit / np.sum(logit, axis=0))
        # Calcul de l'erreur et de la perte:
        pred_to_labels = [np.argmax(softmax_outputs[i]) for i in range(len(softmax_outputs))]
        acc = np.mean(pred_to_labels == y)
        loss = np.mean(losses)
        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################
        return acc, loss

    def momentum_update(self, w, dw, lr, mu):
        """
        Compute momentum as in : http://cs231n.github.io/neural-networks-3/#sgd

        In our case, variable v is stored in self.momentum_cache_v_prev.

        The resulting w is put in parameter "w" which is passed by reference

        Returns nothing
        """
        # self.net.parameters possède 2 numpy
        # donc self.net.parameters[layer] est un type valeur !
        v_prev = self.momentum_cache_v_prev[id(w)]
        layer = self.net.parameters.index(w)
        #############################################################################
        # Update v and w                                                            #
        #############################################################################
        v = mu*v_prev + dw
        w = w - lr*v
        #############################################################################
        # Update w new value                                                        #
        #############################################################################
        # self.net.parameters[layer] = w
        if layer == 0:
            self.net.layer1.W = w
        else:
            self.net.layer2.W = w
        self.momentum_cache_v_prev[id(w)] = v
        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################


class TwoLayerNet(object):
    """
    This class encodes a network with two layers or parameters : one between the input layer
     and the hidden layer and one between the hidden layer and the output layer
    """

    def __init__(self, in_size, hidden_size, num_classes, activation='relu', l2_r=0.0):
        self.in_size = in_size
        self.num_classes = num_classes
        self.l2_reg = l2_r
        self.layer1 = DenseLayer(in_size, hidden_size, activation=activation)
        self.layer2 = DenseLayer(hidden_size, num_classes)

    def reinit(self):
        self.layer1.reinit()
        self.layer2.reinit()

    def forward(self, x):
        x1 = self.layer1.forward(x)
        x2 = self.layer2.forward(x1)
        return x2

    def backward_(self, dloss_dscores):
        dx = self.layer2.backward(dloss_dscores, self.l2_reg)
        self.layer1.backward(dx, self.l2_reg)

    def forward_backward(self, x, y):
        self.layer1.zero_grad()
        self.layer2.zero_grad()

        scores = self.forward(x)
        loss, dscores = self.cross_entropy_loss(scores, y)
        self.backward_(dscores)
        return loss

    @property
    def parameters(self):
        return [self.layer1.W, self.layer2.W]

    @property
    def gradients(self):
        return [self.layer1.dW, self.layer2.dW]

    def cross_entropy_loss(self, scores, y):
        """
        Cross-entropy loss function and gradient with respect to the score
        C.f. Eq.(4.104 to 4.109) of Bishop book.

        Inputs:
        - scores: output of the network before the softmax.  A numpy array of shape (C) (C is the number of classes).
        - y: training label as an integer
        Returns a tuple of:
        - loss as single float
        - gradient with respect to weights; an array of same shape as W1 and W2
        """
        #############################################################################
        #       Compute the softmax loss and its gradient.                          #
        # Store the loss in loss and the gradient in dW.                            #
        # 1- Compute softmax => eq.(4.104) or eq.(5.25) Bishop                      #
        # 2- Compute cross-entropy loss => eq.(4.108)                               #
        # 3- Dont forget the regularization!                                        #
        # 4- Compute gradient with respect to the score => eq.(4.104) with phi_n=1  #
        #############################################################################
        # On transforme notre label en one-hot-vector:
        tmp = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        y = tmp[y]
        # 1-
        logit = np.exp(scores)
        output_softmax = logit / np.sum(logit, axis=0)
        # 2,3-:
        loss = -np.sum(y*np.log(output_softmax)) + self.l2_reg*(np.sum(self.layer1.W**2) + np.sum(self.layer2.W**2))
        # 4- Gradient:
        # 4:
        # Derivée de l'activation:
        dloss = np.array(output_softmax-np.array(y))
        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################
        return loss, dloss


class DenseLayer(object):
    """
    This class encodes a layer (could be the hidden layer or the output layer)
    """
    def __init__(self, in_size, out_size, activation=None):
        self.activation = activation  # Note, 'relu', or 'sigmoid'
        self.W = None
        self.dW = None
        self.in_size = in_size # number of input neurons
        self.out_size = out_size # number of output neurons
        self.reinit()

        self.last_x = None
        self.last_activ = None

    def reinit(self):
        self.W = np.random.randn(self.in_size + 1, self.out_size) * np.sqrt(0.1 / (self.in_size + self.out_size))
        self.dW = np.zeros_like(self.W)

    def zero_grad(self):
        self.dW.fill(0.0)

    def forward(self, x):
        """
        Computer forward pass for 1 layer :

                f=h(dot(W,x))

        where h is the layer's activation function.

        Inputs:
        - x: A numpy array of shape (D)
        Returns a tuple of:
        - f: a floating point value
        """
        #############################################################################
        #       Compute forward pass.  Do not forget to add 1 to x in case of bias  #
        # C.f. function augment(x)                                                  #
        #############################################################################
        x = augment(x)
        f = np.dot(self.W.T, x)
        if self.activation == 'relu':
            f = np.array([f[i] if f[i] > 0 else 0 for i in range(len(f))])
        elif self.activation == 'sigmoid':
            f = sigmoid(f)
        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################
        self.last_x = x
        self.last_activ = f
        return f

    def backward(self, dnext_dout, l2_reg):
        if self.activation == 'sigmoid':
            dout_dsigmoid = self.last_activ * (1.0 - self.last_activ)
            dnext_dsigmoid = dnext_dout * dout_dsigmoid  # type:np.ndarray
            dnext_dW = self.last_x[:, np.newaxis] * dnext_dsigmoid[np.newaxis, :]
            dnext_dX = dnext_dsigmoid.dot(self.W.T)
        elif self.activation == 'relu':
            dout_drelu = self.last_activ != 0.0
            dnext_drelu = dnext_dout * dout_drelu  # type:np.ndarray
            dnext_dW = self.last_x[:, np.newaxis] * dnext_drelu[np.newaxis, :]
            dnext_dX = dnext_drelu.dot(self.W.T)
        else:
            dnext_dW = self.last_x[:, np.newaxis].dot(dnext_dout[np.newaxis, :])
            dnext_dX = dnext_dout.dot(self.W.T)
        dnext_dX = dnext_dX[:-1]  # discard the gradient wrt the 1.0 of homogeneous coord
        self.dW += dnext_dW
        self.dW += l2_reg * self.W  # add regul. gradient
        return dnext_dX


def augment(x):
    if len(x.shape) == 1:
        return np.concatenate([x, [1.0]])
    else:
        return np.concatenate([x, np.ones((len(x), 1))], axis=1)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
